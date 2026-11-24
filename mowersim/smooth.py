"""Paths the steering can follow: curvature that changes at a bounded rate.

The planners join straight lines and arcs, so the curvature jumps where they
meet, and the steering would have to turn instantly. A steering motor turns at
most ``max_steer_rate_deg`` per second. At speed v the curvature
kappa = tan(delta) / L then changes with the distance s at most at the rate

    |d kappa / d s| = |d delta / d t| / (L cos^2 delta v)  >=  omega / (L v)

so curvature ramps of slope sigma = omega / (L v) (clothoids) are always
feasible (course, chapter 10). Where the mower stops to change direction it
can turn the steering while standing, and the curvature may jump.

``smooth`` replaces every jump of curvature while moving by a ramp, and keeps
the path anchored: it cuts the trajectory at the middle of its long straight
lines, and in each run between two anchors it adjusts the lengths of the
segments and the curvature of the arcs (Levenberg-Marquardt) so that the run
still ends exactly at the next anchor. A ramp is represented by short arcs of
increasing curvature, so the result is still a list of Segments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Pose
from .kinematics import Segment, Trajectory, advance
from .mower import Mower
from .paths import pify

RAMP_PIECES = 12


def sharpness(mower: Mower, reverse: bool = False) -> float:
    """Largest safe rate of change of curvature per metre, sigma = omega / (L v),
    at the speed of arcs (forward) or of reverse."""
    v = mower.v_reverse if reverse else mower.v_turn
    return math.radians(mower.max_steer_rate_deg) / (mower.wheelbase * v)


def ramp(k0: float, k1: float, sigma: float, reverse: bool, mow: bool,
         scale: float = 1.0) -> list[Segment]:
    """A clothoid from curvature k0 to k1 (slope sigma), as short arcs taking the
    curvature at the middle of each piece. ``scale`` < 1 shortens it (used only
    on the way to the solution, see smooth_run)."""
    if abs(k1 - k0) < 1e-12:
        return []
    length = scale * abs(k1 - k0) / sigma
    h = length / RAMP_PIECES
    return [Segment(k0 + (k1 - k0) * (i + 0.5) / RAMP_PIECES, h, reverse, mow)
            for i in range(RAMP_PIECES)]


def layout(curv, lengths, rev, mow, mower: Mower, k_in: float | None = 0.0,
           k_out: float | None = 0.0, scale: float = 1.0):
    """The pieces of a run with ramps, as arrays (curvature, signed length,
    reverse, blades). Each change of curvature between two segments driven in
    the same direction becomes a ramp centred on the junction: it takes half its
    length from the end of the segment before and half from the start of the
    segment after (as much as they have), so that it turns the heading by as
    much as the corner it replaces. A ramp from k_in (None: none, the mower
    starts standing) is added before the first segment, one to k_out after the
    last."""
    n = len(curv)
    ramp_len = np.zeros(n + 1)           # ramp_len[i]: ramp before segment i
    for i in range(n + 1):
        if i == 0:
            k0, k1, r = k_in, (curv[0] if n else None), (rev[0] if n else False)
        elif i == n:
            k0, k1, r = curv[-1], k_out, rev[-1]
        else:
            if rev[i] != rev[i - 1]:
                continue                 # a stop: the steering turns while standing
            k0, k1, r = curv[i - 1], curv[i], rev[i]
        if k0 is None or k1 is None or abs(k1 - k0) < 1e-12:
            continue
        ramp_len[i] = scale * abs(k1 - k0) / sharpness(mower, r)
    take_in = ramp_len[:n] / 2
    take_out = ramp_len[1:] / 2
    take_in[0] = 0.0                     # the end ramps are added (the run starts and
    take_out[-1] = 0.0                   # ends with the curvature k_in and k_out)
    const = np.maximum(np.asarray(lengths, float) - take_in - take_out, 0.0)
    frac = (np.arange(RAMP_PIECES) + 0.5) / RAMP_PIECES
    ks, ls, rs, ms = [], [], [], []

    def add_ramp(i, k0, k1, r, m):
        h = ramp_len[i] / RAMP_PIECES
        ks.append(k0 + (k1 - k0) * frac)
        ls.append(np.full(RAMP_PIECES, -h if r else h))
        rs.append(np.full(RAMP_PIECES, r))
        ms.append(np.full(RAMP_PIECES, m))
    for i in range(n):
        if ramp_len[i] > 0:
            k0 = k_in if i == 0 else curv[i - 1]
            add_ramp(i, k0, curv[i], rev[i], mow[i] and (i == 0 or mow[i - 1]))
        ks.append(np.array([curv[i]]))
        ls.append(np.array([-const[i] if rev[i] else const[i]]))
        rs.append(np.array([rev[i]]))
        ms.append(np.array([mow[i]]))
    if n and ramp_len[n] > 0:
        add_ramp(n, curv[-1], k_out, rev[-1], mow[-1])
    if not ks:
        return np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)
    return np.concatenate(ks), np.concatenate(ls), np.concatenate(rs), np.concatenate(ms)


def build(targets: list[Segment], mower: Mower, k_in: float | None = 0.0,
          k_out: float | None = 0.0, scale: float = 1.0) -> list[Segment]:
    """The segments of a run with centred ramps (see layout)."""
    k, ln, r, m = layout([sg.curvature for sg in targets], [sg.length for sg in targets],
                         [sg.reverse for sg in targets], [sg.mow for sg in targets],
                         mower, k_in, k_out, scale)
    return [Segment(float(a), float(abs(b)), bool(c), bool(d))
            for a, b, c, d in zip(k, ln, r, m) if abs(b) > 1e-12]


def pose_after(start: Pose, k: np.ndarray, ln: np.ndarray) -> tuple[float, float, float]:
    """End pose after the arcs (k, signed length), vectorised."""
    th_after = start.theta + np.cumsum(k * ln)
    th_before = th_after - k * ln
    curved = np.abs(k) > 1e-12
    ksafe = np.where(curved, k, 1.0)
    dx = np.where(curved, (np.sin(th_after) - np.sin(th_before)) / ksafe,
                  ln * np.cos(th_before))
    dy = np.where(curved, -(np.cos(th_after) - np.cos(th_before)) / ksafe,
                  ln * np.sin(th_before))
    th = th_after[-1] if len(k) else start.theta
    return start.x + float(dx.sum()), start.y + float(dy.sum()), float(th)


def end_pose(start: Pose, segs: list[Segment]) -> Pose:
    p = start
    for sg in segs:
        p = advance(p, sg.curvature, sg.signed_length)
    return p


@dataclass
class RunResult:
    segments: list[Segment]
    ok: bool
    error: float


def smooth_run(start: Pose, targets: list[Segment], goal: Pose, mower: Mower,
               k_in: float | None = 0.0, k_out: float | None = 0.0,
               steps: int = 5, iterations: int = 30) -> RunResult:
    """Ramps between the segments of a run, with the lengths of the segments and
    the curvatures of its arcs adjusted so that the run still ends at ``goal``.

    Solved by continuation: the ramps grow from nothing (where the original run
    is the solution) to their full length in ``steps`` stages, each solved by
    Levenberg-Marquardt from the solution of the previous stage."""
    kmax = mower.max_curvature
    arcs = [i for i, sg in enumerate(targets) if sg.curvature != 0]
    n = len(targets)

    def make(z, scale):
        segs = [Segment(sg.curvature, max(z[i], 0.0), sg.reverse, sg.mow)
                for i, sg in enumerate(targets)]
        for j, i in enumerate(arcs):
            k = float(np.clip(z[n + j], -kmax, kmax))
            segs[i] = Segment(k, segs[i].length, segs[i].reverse, segs[i].mow)
        return build(segs, mower, k_in, k_out, scale)

    rev = [sg.reverse for sg in targets]
    mows = [sg.mow for sg in targets]

    def residual(z, scale):
        curv = [sg.curvature for sg in targets]
        for j, i in enumerate(arcs):
            curv[i] = float(np.clip(z[n + j], -kmax, kmax))
        k, ln, _, _ = layout(curv, np.maximum(z[:n], 0.0), rev, mows, mower, k_in, k_out,
                             scale)
        x, y, th = pose_after(start, k, ln)
        return np.array([x - goal.x, y - goal.y, pify(th - goal.theta)])

    z = np.array([sg.length for sg in targets] + [targets[i].curvature for i in arcs])
    weight = np.array([1.0] * n + [0.3] * len(arcs))    # curvature moves less
    for scale in np.linspace(1.0 / steps, 1.0, steps):
        r = residual(z, scale)
        lam = 1e-3
        for _ in range(iterations):
            if np.linalg.norm(r) < 1e-7:
                break
            J = np.empty((3, len(z)))
            for k in range(len(z)):
                dz = np.zeros_like(z)
                dz[k] = 1e-6
                J[:, k] = (residual(z + dz, scale) - r) / 1e-6
            Jw = J * weight
            improved = False
            while lam < 1e6:
                step = -weight * np.linalg.solve(Jw.T @ Jw + lam * np.eye(len(z)), Jw.T @ r)
                z1 = z + step
                z1[:n] = np.maximum(z1[:n], 0.0)
                z1[n:] = np.clip(z1[n:], -kmax, kmax)
                r1 = residual(z1, scale)
                if np.linalg.norm(r1) < np.linalg.norm(r):
                    z, r, lam, improved = z1, r1, max(lam / 3, 1e-9), True
                    break
                lam *= 10
            if not improved:
                break
    err = float(np.linalg.norm(residual(z, 1.0)))
    return RunResult(make(z, 1.0), err < 1e-3, err)


@dataclass
class SmoothStats:
    runs: int                        # runs between two anchors that need ramps
    smoothed: int                    # ... smoothed completely
    failed: int                      # parts kept as they were (no solution found)
    partly: int = 0                  # runs smoothed in parts, some kept as they were


def _split(run: list[Segment], anchor: float) -> list[list[Segment]]:
    """Cut a list of segments at the middle of its straight segments at least
    ``anchor`` long; the halves end one part and start the next."""
    parts: list[list[Segment]] = [[]]
    for i, sg in enumerate(run):
        inner = 0 < i < len(run) - 1
        if sg.curvature == 0 and sg.length >= anchor and (inner or anchor >= 1.0):
            half = Segment(0.0, sg.length / 2, sg.reverse, sg.mow)
            parts[-1].append(half)
            parts.append([half])
        else:
            parts[-1].append(sg)
    return [p for p in parts if p]


def smooth(traj: Trajectory, mower: Mower, checker=None, anchor: float = 1.5,
           words: int = 12, fine_anchor: float = 0.3) -> tuple[Trajectory, SmoothStats]:
    """Rate-limited version of a whole trajectory. Anchors: the middle of every
    straight segment at least ``anchor`` long (curvature 0 there, pose kept),
    and the two ends. For each run between two anchors, the planner tries the
    run itself, then the other Reeds-Shepp words between the ends of its
    half-straights, fastest first (up to ``words``; only for runs of at most six
    segments and 15 m, or with the blades off: turns and transfers, not the
    curved parts of a headland), also from 0.5, 1 and 1.5 m earlier on the
    incoming straight to as much later on the outgoing one, and keeps the first
    one whose smoothed version reaches the anchor and, if a collision
    ``checker`` is given, stays on the lawn. A long run that fails is cut again
    at its straight segments at least ``fine_anchor`` long, and the parts are
    smoothed one by one. A part with no solution is kept as it was."""
    stats = SmoothStats(0, 0, 0)
    out: list[Segment] = []
    pose = traj.start
    runs = _split([sg for sg in traj.segments if sg.length > 1e-9], anchor)
    for k, run in enumerate(runs):
        k_in = None if k == 0 else 0.0
        k_out = None if k == len(runs) - 1 else 0.0
        segs = _smooth_part(pose, run, mower, checker, k_in, k_out, words, fine_anchor,
                            stats)
        out += segs
        pose = end_pose(pose, run)
    return Trajectory(traj.start, out), stats


def _smooth_part(pose, run, mower, checker, k_in, k_out, words, fine_anchor, stats,
                 depth: int = 0) -> list[Segment]:
    from .paths import candidates, path_time
    m = mower
    goal = end_pose(pose, run)
    needs = any(a.curvature != b.curvature and a.reverse == b.reverse
                for a, b in zip(run, run[1:]))
    if not needs:
        return run
    if depth == 0:
        stats.runs += 1
    head = [] if k_in is None else run[:1]
    tail = [] if k_out is None else run[-1:]
    mid = run[len(head):len(run) - len(tail)]
    attempts = [(run, 5), (run, 20)]
    # other words only for a turn or a transfer, never to cut short a stretch
    # that mows along a border (many short segments); the words may start
    # earlier on the incoming straight and end later on the outgoing one
    if mid and (len(mid) <= 6 or not any(sg.mow for sg in mid)) \
            and sum(sg.length for sg in mid) <= 15.0:
        mow = all(sg.mow for sg in mid)
        room = min([sg.length for sg in head + tail] or [0.0])
        for b in np.arange(0.0, min(room - 0.1, 2.0), 0.5):
            h2 = [Segment(0.0, sg.length - b, sg.reverse, sg.mow) for sg in head]
            t2 = [Segment(0.0, sg.length - b, sg.reverse, sg.mow) for sg in tail]
            p1 = end_pose(pose, h2)
            p2 = end_pose(pose, head + mid + ([Segment(0.0, b, tail[0].reverse)]
                                              if tail else []))
            alt = sorted(candidates(p1, p2, m.min_radius),
                         key=lambda w: path_time(w.segments(m.min_radius), m.v_mow,
                                                 m.v_turn, m.v_reverse, m.t_shift))
            attempts += [(h2 + w.segments(m.min_radius, mow) + t2, 5) for w in alt[:words]]
    collided = False
    for n_att, (att, steps) in enumerate(attempts):
        if n_att == 1 and collided:
            continue                       # it converged but collides: finer stages
        res = smooth_run(pose, att, goal, mower, k_in, k_out, steps=steps)  # won't help
        if res.ok and (checker is None or checker.free(pose, res.segments)):
            if depth == 0:
                stats.smoothed += 1
            return res.segments
        collided = n_att == 0 and res.ok
    parts = _split(run, fine_anchor) if depth == 0 and len(run) > 6 else [run]
    if len(parts) > 1:
        out, p, ok = [], pose, True
        for j, part in enumerate(parts):
            ki = k_in if j == 0 else 0.0
            ko = k_out if j == len(parts) - 1 else 0.0
            before = stats.failed
            seg = _smooth_part(p, part, mower, checker, ki, ko, words, fine_anchor, stats,
                               depth + 1)
            ok = ok and stats.failed == before
            out += seg
            p = end_pose(p, part)
        if ok:
            stats.smoothed += 1
        else:
            stats.partly += 1
        return out
    stats.failed += 1
    return run


def steering_profile(segs: list[Segment], mower: Mower):
    """Time, steering angle and its rate along a path (stops not counted)."""
    t, d = [0.0], [mower.steering_angle(segs[0].curvature) if segs else 0.0]
    for sg in segs:
        v = mower.v_reverse if sg.reverse else (mower.v_mow if sg.curvature == 0
                                                 else mower.v_turn)
        t.append(t[-1] + sg.length / v)
        d.append(mower.steering_angle(sg.curvature))
    return np.array(t), np.degrees(np.array(d))


def curvature_jumps(segs: list[Segment], mower: Mower, step: float = 0.15) -> int:
    """Number of places where, while moving, the curvature steps by more than the
    steering can change it over ``step`` metres (the short arcs of a ramp, at
    most 2 kappa_max / sigma / 12 = 0.14 m long, step
    by less; a change at a stop is allowed)."""
    bad = 0
    for a, b in zip(segs, segs[1:]):
        if a.reverse != b.reverse or a.length < 1e-9 or b.length < 1e-9:
            continue
        if abs(b.curvature - a.curvature) > sharpness(mower, b.reverse) * step:
            bad += 1
    return bad
