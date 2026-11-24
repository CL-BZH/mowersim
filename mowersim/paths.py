"""Shortest paths for a car-like vehicle: Dubins (forward only) and Reeds-Shepp.

Both work in normalised coordinates: the start pose is moved to the origin,
heading along x, and lengths are divided by the minimum turning radius R.
A path is then a *word* of at most five letters, L (left arc), R (right arc)
and S (straight), each with a signed length: radians for an arc, multiples of R
for a straight line, negative when driving in reverse.

* Dubins (1957): the shortest forward path between two poses, with curvature
  at most 1/R, is one of the 6 words LSL, RSR, LSR, RSL, RLR, LRL.
* Reeds and Shepp (1990): with reverse allowed, the shortest path is one of
  48 words, with at most 5 segments and 2 changes of direction. The 48 words
  are obtained from 12 base formulas and two symmetries:
    - time flip: drive the path backwards (all lengths change sign),
      which solves the problem for the goal (-x, y, -phi);
    - reflection: swap L and R, which solves it for (x, -y, -phi).

Every candidate is checked by driving it (kinematics.advance) and comparing
the end pose with the goal, so a word is only returned if it really gets
there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose
from .kinematics import Segment, advance

PI = math.pi
TOL = 1e-6


def mod2pi(a: float) -> float:
    """Angle in [0, 2 pi)."""
    v = math.fmod(a, 2 * PI)
    return v + 2 * PI if v < 0 else v


def pify(a: float) -> float:
    """Angle in (-pi, pi]."""
    v = mod2pi(a)
    return v - 2 * PI if v > PI else v


def polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


@dataclass(frozen=True)
class Word:
    """A path in normalised units: letters and signed lengths."""
    letters: str
    lengths: tuple[float, ...]

    @property
    def length(self) -> float:
        return sum(abs(t) for t in self.lengths)

    def segments(self, radius: float, mow: bool = False) -> list[Segment]:
        out = []
        for letter, t in zip(self.letters, self.lengths):
            if abs(t) < 1e-12:
                continue
            k = {"L": 1.0 / radius, "R": -1.0 / radius, "S": 0.0}[letter]
            out.append(Segment(k, abs(t) * radius, reverse=t < 0, mow=mow))
        return out


def normalise(start: Pose, goal: Pose, radius: float) -> tuple[float, float, float]:
    """Goal pose in the start's frame, lengths divided by the turning radius."""
    dx, dy = goal.x - start.x, goal.y - start.y
    c, s = math.cos(start.theta), math.sin(start.theta)
    return (c * dx + s * dy) / radius, (-s * dx + c * dy) / radius, pify(goal.theta - start.theta)


def drive(start: Pose, segments: list[Segment]) -> Pose:
    pose = start
    for seg in segments:
        pose = advance(pose, seg.curvature, seg.signed_length)
    return pose


def reaches(start: Pose, goal: Pose, segments: list[Segment], tol: float = 1e-6) -> bool:
    end = drive(start, segments)
    return (math.hypot(end.x - goal.x, end.y - goal.y) < tol * 10
            and abs(pify(end.theta - goal.theta)) < tol * 10)


# ===================================================================== Dubins
def dubins_words(x: float, y: float, phi: float) -> list[Word]:
    """The 6 Dubins words from (0, 0, 0) to (x, y, phi), normalised (R = 1).

    Classic formulation: d is the distance to the goal, and alpha, beta the start
    and goal headings measured from the line joining them.
    """
    d, th = polar(x, y)
    a, b = mod2pi(-th), mod2pi(phi - th)
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    cab = math.cos(a - b)
    out = []

    tmp = 2 + d * d - 2 * cab + 2 * d * (sa - sb)                       # LSL
    if tmp >= 0:
        q = math.atan2(cb - ca, d + sa - sb)
        out.append(Word("LSL", (mod2pi(-a + q), math.sqrt(tmp), mod2pi(b - q))))
    tmp = 2 + d * d - 2 * cab + 2 * d * (sb - sa)                       # RSR
    if tmp >= 0:
        q = math.atan2(ca - cb, d - sa + sb)
        out.append(Word("RSR", (mod2pi(a - q), math.sqrt(tmp), mod2pi(-b + q))))
    tmp = -2 + d * d + 2 * cab + 2 * d * (sa + sb)                      # LSR
    if tmp >= 0:
        p = math.sqrt(tmp)
        q = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(Word("LSR", (mod2pi(-a + q), p, mod2pi(-b + q))))
    tmp = d * d - 2 + 2 * cab - 2 * d * (sa + sb)                       # RSL
    if tmp >= 0:
        p = math.sqrt(tmp)
        q = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(Word("RSL", (mod2pi(a - q), p, mod2pi(b - q))))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8                 # RLR
    if abs(tmp) <= 1:
        p = mod2pi(2 * PI - math.acos(tmp))
        t = mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
        out.append(Word("RLR", (t, p, mod2pi(a - b - t + p))))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8                 # LRL
    if abs(tmp) <= 1:
        p = mod2pi(2 * PI - math.acos(tmp))
        t = mod2pi(-a - math.atan2(ca - cb, d + sa - sb) + p / 2)
        out.append(Word("LRL", (t, p, mod2pi(b - a - t + p))))
    return out


# =============================================================== Reeds-Shepp
# Base formulas, after Reeds & Shepp (1990), in the form used by OMPL and
# PythonRobotics. Each returns the signed lengths (t, u, v, ...) or None.
def _LSL(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1 + math.cos(phi))
    if t >= -TOL:
        v = mod2pi(phi - t)
        if v >= -TOL:
            return "LSL", (t, u, v)


def _LSR(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1 - math.cos(phi))
    if u1 * u1 >= 4:
        u = math.sqrt(u1 * u1 - 4)
        t = mod2pi(t1 + math.atan2(2, u))
        v = mod2pi(t - phi)
        return "LSR", (t, u, v)


def _LxRxL(x, y, phi):                     # L+ R- L+
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 <= 4:
        a = math.acos(u1 / 4)
        t = mod2pi(a + th + PI / 2)
        u = mod2pi(PI - 2 * a)
        v = mod2pi(phi - t - u)
        return "LRL", (t, -u, v)


def _LxRL(x, y, phi):                      # L+ R- L-
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 <= 4:
        a = math.acos(u1 / 4)
        t = mod2pi(a + th + PI / 2)
        u = mod2pi(PI - 2 * a)
        v = mod2pi(-phi + t + u)
        return "LRL", (t, -u, -v)


def _LRxL(x, y, phi):                      # L+ R+ L-
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, th = polar(xi, eta)
    if 0 < u1 <= 4:
        u = math.acos(1 - u1 * u1 / 8)
        a = math.asin(max(-1.0, min(1.0, 2 * math.sin(u) / u1)))
        t = mod2pi(-a + th + PI / 2)
        v = mod2pi(t - u - phi)
        return "LRL", (t, u, -v)


def _LRxLR(x, y, phi):                     # L+ R+ L- R-
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 <= 2:
        a = math.acos((u1 + 2) / 4)
        t = mod2pi(th + a + PI / 2)
        u = mod2pi(a)
        v = mod2pi(phi - t + 2 * u)
        return "LRLR", (t, u, -u, -v)


def _LxRLxR(x, y, phi):                    # L+ R- L- R+
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, th = polar(xi, eta)
    u2 = (20 - u1 * u1) / 16
    if 0 <= u2 <= 1 and u1 > 0:
        u = math.acos(u2)
        a = math.asin(max(-1.0, min(1.0, 2 * math.sin(u) / u1)))
        t = mod2pi(th + a + PI / 2)
        v = mod2pi(t - phi)
        return "LRLR", (t, -u, -u, v)


def _LxR90SL(x, y, phi):                   # L+ R-(pi/2) S- L-
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 >= 2:
        r = math.sqrt(u1 * u1 - 4)
        u = r - 2
        a = math.atan2(2, r)
        t = mod2pi(th + a + PI / 2)
        v = mod2pi(t - phi + PI / 2)
        if u >= -TOL:
            return "LRSL", (t, -PI / 2, -u, -v)


def _LSR90xL(x, y, phi):                   # L+ S+ R+(pi/2) L-
    xi, eta = x - math.sin(phi), y - 1 + math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 >= 2:
        r = math.sqrt(u1 * u1 - 4)
        u = r - 2
        a = math.atan2(r, 2)
        t = mod2pi(th - a + PI / 2)
        v = mod2pi(t - phi - PI / 2)
        if u >= -TOL:
            return "LSRL", (t, u, PI / 2, -v)


def _LxR90SR(x, y, phi):                   # L+ R-(pi/2) S- R-
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 >= 2:
        t = mod2pi(th + PI / 2)
        u = u1 - 2
        v = mod2pi(phi - t - PI / 2)
        return "LRSR", (t, -PI / 2, -u, -v)


def _LSL90xR(x, y, phi):                   # L+ S+ L+(pi/2) R-
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 >= 2:
        t = mod2pi(th)
        u = u1 - 2
        v = mod2pi(phi - t - PI / 2)
        return "LSLR", (t, u, PI / 2, -v)


def _LxR90SL90xR(x, y, phi):               # L+ R-(pi/2) S- L-(pi/2) R+
    xi, eta = x + math.sin(phi), y - 1 - math.cos(phi)
    u1, th = polar(xi, eta)
    if u1 >= 4:
        r = math.sqrt(u1 * u1 - 4)
        u = r - 4
        a = math.atan2(2, r)
        t = mod2pi(th + a + PI / 2)
        v = mod2pi(t - phi)
        if u >= -TOL:
            return "LRSLR", (t, -PI / 2, -u, -PI / 2, v)


BASE_FORMULAS = (_LSL, _LSR, _LxRxL, _LxRL, _LRxL, _LRxLR, _LxRLxR, _LxR90SL, _LSR90xL,
                 _LxR90SR, _LSL90xR, _LxR90SL90xR)
_SWAP = str.maketrans("LR", "RL")


def reeds_shepp_words(x: float, y: float, phi: float) -> list[Word]:
    """All candidate Reeds-Shepp words from (0, 0, 0) to (x, y, phi), R = 1."""
    out = []
    for f in BASE_FORMULAS:
        for flip, reflect in ((False, False), (True, False), (False, True), (True, True)):
            gx = -x if flip else x
            gy = -y if reflect else y
            gphi = -phi if flip != reflect else phi
            res = f(gx, gy, gphi)
            if res is None:
                continue
            letters, lengths = res
            if flip:
                lengths = tuple(-t for t in lengths)
            if reflect:
                letters = letters.translate(_SWAP)
            out.append(Word(letters, tuple(lengths)))
    return out


# ================================================================ public API
def candidates(start: Pose, goal: Pose, radius: float, reverse: bool = True) -> list[Word]:
    """All words that really drive from start to goal (checked by integration),
    shortest first. ``reverse=False`` gives the Dubins words."""
    x, y, phi = normalise(start, goal, radius)
    words = reeds_shepp_words(x, y, phi) if reverse else dubins_words(x, y, phi)
    good = {}
    for w in words:
        if reaches(start, goal, w.segments(radius)):
            key = (w.letters, tuple(round(t, 9) for t in w.lengths))
            good[key] = w
    return sorted(good.values(), key=lambda w: w.length)


def shortest(start: Pose, goal: Pose, radius: float, reverse: bool = True,
             mow: bool = False) -> list[Segment]:
    """Segments of the shortest path (Reeds-Shepp, or Dubins if reverse=False)."""
    words = candidates(start, goal, radius, reverse)
    if not words:
        raise ValueError("no path found")  # does not happen for valid inputs
    return words[0].segments(radius, mow)


def path_time(segments: list[Segment], v_straight: float, v_turn: float, v_reverse: float,
              t_shift: float) -> float:
    """Driving time: straight and arc speeds forward, one speed in reverse, plus a fixed
    time at every change of direction."""
    t = 0.0
    for seg in segments:
        v = v_reverse if seg.reverse else (v_straight if seg.curvature == 0 else v_turn)
        t += seg.length / v
    moving = [seg.reverse for seg in segments if seg.length > 0]
    return t + t_shift * sum(a != b for a, b in zip(moving, moving[1:]))


def fastest(start: Pose, goal: Pose, mower, reverse: bool = True,
            mow: bool = False) -> list[Segment]:
    """Among the Reeds-Shepp (or Dubins) candidate words, the one that takes the least
    time for this mower. The shortest path is not always the fastest: a change of
    direction costs ``t_shift`` seconds and reverse is slower. (This picks the best of
    the candidate words; it is not a time-optimal planner.)"""
    words = candidates(start, goal, mower.min_radius, reverse)
    best = min(words, key=lambda w: path_time(w.segments(mower.min_radius), mower.v_mow,
                                              mower.v_turn, mower.v_reverse, mower.t_shift))
    return best.segments(mower.min_radius, mow)
