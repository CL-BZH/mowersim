"""Driving a planned path: a simulated mower that follows it with pure pursuit.

The real mower does not drive the plan exactly. It knows its pose only from a
localisation system (satellite positioning with corrections, odometry), with
some error; its steering turns at a limited rate; and a controller turns the
difference between where it is and where the path goes into a steering
command. This module simulates that loop (course, chapter 11):

* the reference path is cut into pieces of constant direction (forward or
  reverse) at the changes of direction, where the mower stops;
* on each piece the controller compares the estimated pose with the closest
  point of the path and asks for the curvature of the path there, corrected by
  the lateral and heading errors (feedback, a critically damped correction
  over a distance ``settle``); or, as an alternative, it aims at the point of
  the path a lookahead distance ahead and asks for the curvature of the arc
  through it (pure pursuit, Coulter 1992);
* the steering moves towards the command at most at max_steer_rate_deg per
  second; the speed is that of the plan at the closest point;
* the estimated pose is the true pose plus Gaussian noise, drawn afresh at
  every update of the localisation, and carried forward in between by the
  motion the mower knows it made (odometry).

The executed motion is returned as a Trajectory of short arcs, so the
coverage, the collisions and the time are measured as for any plan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Pose
from .kinematics import Segment, Trajectory, advance
from .mower import Mower
from .paths import pify


@dataclass
class Piece:
    """A stretch of the reference driven in one direction."""
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    curvature: np.ndarray
    s: np.ndarray            # distance along the piece
    speed: np.ndarray
    mow: np.ndarray
    reverse: bool


def pieces(traj: Trajectory, mower: Mower, step: float = 0.02) -> list[Piece]:
    smp = traj.sample(step)
    out = []
    rev = smp.reverse
    cut = [0] + [i for i in range(1, len(rev)) if rev[i] != rev[i - 1]] + [len(rev)]
    for a, b in zip(cut, cut[1:]):
        if b - a < 2:
            continue
        sl = slice(a, b)
        k = smp.curvature[sl]
        r = bool(rev[a])
        v = np.where(r, mower.v_reverse, np.where(k == 0, mower.v_mow, mower.v_turn))
        out.append(Piece(smp.x[sl], smp.y[sl], smp.theta[sl], k,
                         smp.s[sl] - smp.s[a], v.astype(float), smp.mow[sl], r))
    return out


@dataclass
class Follower:
    controller: str = "feedback"     # "feedback" or "pure_pursuit"
    settle: float = 1.0              # m: distance over which the feedback corrects an error
    lookahead: float = 0.8           # m, pure pursuit, at the speed of arcs; grows with speed
    noise_pos: float = 0.0           # m, standard deviation of the position error
    noise_heading_deg: float = 0.0   # standard deviation of the heading error
    update: float = 0.1              # s between localisation updates
    dt: float = 0.02                 # s, simulation step
    seed: int = 0


@dataclass
class Run:
    trajectory: Trajectory
    errors: np.ndarray               # distance to the path at each step (m)
    time: float
    piece: np.ndarray                # index of the piece of the path at each step


def follow(traj: Trajectory, mower: Mower, f: Follower = Follower(),
           start: Pose | None = None) -> Run:
    """Drive ``traj`` with the follower ``f``, from ``start`` (default: the start of
    the path)."""
    rng = np.random.default_rng(f.seed)
    omega = math.radians(mower.max_steer_rate_deg)
    dmax = math.radians(mower.max_steer_deg)
    pose = traj.start if start is None else start
    delta = 0.0
    segs: list[Segment] = []
    errors = []
    t = 0.0
    est = pose
    next_update = 0.0
    prev_rev = None
    owner = []
    for p_index, pc in enumerate(pieces(traj, mower)):
        if prev_rev is not None:
            t += mower.t_shift                        # stop: re-steer while standing
            k0 = pc.curvature[0]
            target = math.atan(mower.wheelbase * k0)
            delta = float(np.clip(target, delta - omega * mower.t_shift,
                                  delta + omega * mower.t_shift))
        prev_rev = pc.reverse
        sign = -1.0 if pc.reverse else 1.0
        i = 0
        n = len(pc.s)
        for _ in range(int(4 * (pc.s[-1] + 5) / (0.5 * mower.v_reverse * f.dt))):
            if t >= next_update:                      # a new position fix
                ex = rng.normal(0, f.noise_pos, 2) if f.noise_pos else (0.0, 0.0)
                eh = math.radians(rng.normal(0, f.noise_heading_deg)) \
                    if f.noise_heading_deg else 0.0
                est = Pose(pose.x + ex[0], pose.y + ex[1], pose.theta + eh)
                next_update = t + f.update
            # closest point ahead of the last one (the path is followed in order)
            j = slice(i, min(i + 200, n))
            d2 = (pc.x[j] - est.x) ** 2 + (pc.y[j] - est.y) ** 2
            i = i + int(np.argmin(d2))
            # true distance to the path, for the record
            w = slice(max(i - 100, 0), min(i + 100, n))
            errors.append(float(np.sqrt(((pc.x[w] - pose.x) ** 2
                                         + (pc.y[w] - pose.y) ** 2).min())))
            owner.append(p_index)
            # the end of the piece: stop when it is reached or passed
            tx, ty = math.cos(pc.theta[-1]), math.sin(pc.theta[-1])
            along = ((est.x - pc.x[-1]) * tx + (est.y - pc.y[-1]) * ty) * sign
            if i >= n - 1 or (pc.s[-1] - pc.s[i] < 0.5 and along >= 0):
                break
            v = float(pc.speed[i])
            if f.controller == "pure_pursuit":
                ld = f.lookahead * max(v / mower.v_turn, 1.0)
                k = int(np.searchsorted(pc.s, pc.s[i] + ld))
                if k >= n:                            # beyond the end: extend the last heading
                    extra = pc.s[i] + ld - pc.s[-1]
                    gx = pc.x[-1] + sign * extra * math.cos(pc.theta[-1])
                    gy = pc.y[-1] + sign * extra * math.sin(pc.theta[-1])
                else:
                    gx, gy = pc.x[k], pc.y[k]
                # in the frame of the estimated pose, along the direction of travel
                hx, hy = math.cos(est.theta) * sign, math.sin(est.theta) * sign
                dx, dy = gx - est.x, gy - est.y
                lat = -hy * dx + hx * dy              # to the left of the direction of travel
                k_geo = 2 * lat / max(dx * dx + dy * dy, 1e-6)
            else:
                # errors relative to the closest point, along the direction of travel
                ux, uy = math.cos(pc.theta[i]) * sign, math.sin(pc.theta[i]) * sign
                e_y = -uy * (est.x - pc.x[i]) + ux * (est.y - pc.y[i])
                e_th = pify(est.theta - pc.theta[i])
                k_ref = float(pc.curvature[i]) * sign
                k_geo = k_ref - e_y / f.settle ** 2 - 2 * math.sin(e_th) / f.settle
            kappa = k_geo * sign                      # wheel angle: same sign either way
            target = math.atan(mower.wheelbase * kappa)
            target = max(-dmax, min(dmax, target))
            delta += max(-omega * f.dt, min(omega * f.dt, target - delta))
            curv = math.tan(delta) / mower.wheelbase
            seg = Segment(curv, v * f.dt, pc.reverse, bool(pc.mow[i]))
            segs.append(seg)
            pose = advance(pose, curv, seg.signed_length)
            est = advance(est, curv, seg.signed_length)   # odometry between two fixes
            t += f.dt
    out = Trajectory(traj.start if start is None else start, segs)
    return Run(out, np.array(errors), t, np.array(owner))


def heading_error(p: Pose, q: Pose) -> float:
    return abs(pify(p.theta - q.theta))
