"""Main passes (swaths) across a region, and the turns between them.

The swaths are the lines of the deck's centre, parallel to a direction theta
and ``s`` apart, clipped to the region still to mow (the interior left by the
headland). The rear axle runs ``d`` behind the deck's centre. Consecutive
passes in the mowing order are joined by the fastest Reeds-Shepp turn that
keeps the mower on the lawn; if none exists, both ends are pulled back along
their passes until one does.

Two orders (course, chapter 7):

* ``adjacent``: passes 0, 1, 2, ... joined by three-point turns, since the
  next pass is closer than twice the turning radius;
* ``skip``: the order of least total turn time; it jumps over a few passes so
  that every turn is a forward U-turn, but needs a wider headland.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from ..geometry import Lawn, Pose
from ..kinematics import Segment, Trajectory
from ..mower import Mower
from ..paths import path_time
from .common import Checker, Connection, connect, polygons, unit


@dataclass
class Swath:
    a: np.ndarray            # where the deck's centre enters the region
    b: np.ndarray            # where it leaves it (a -> b along theta)
    q: float                 # lateral position (along the normal of theta)


def swaths(region, theta: float, mower: Mower, overlap: float = 0.05,
           inset: float | None = None) -> list[Swath]:
    """Deck-centre lines clipped to the region, sorted by lateral position. Each
    line stops ``inset`` before the edge of the region (by default the blade
    radius r: the front of the blades then reaches the edge)."""
    inset = mower.blade_radius if inset is None else inset
    s = mower.cutting_width - overlap
    w = mower.cutting_width
    u, n = unit(theta), unit(theta + math.pi / 2)
    out = []
    for poly in polygons(region):
        pts = np.asarray(poly.exterior.coords)
        proj_u, proj_n = pts @ u, pts @ n
        q0, q1 = proj_n.min(), proj_n.max()
        lo, hi = proj_u.min() - 1, proj_u.max() + 1
        first, last = q0 + w / 2 - overlap, q1 - w / 2 + overlap
        if last < first:                                  # narrower than one pass
            qs = [(q0 + q1) / 2]
        else:
            count = math.ceil((last - first) / s - 1e-9) + 1
            qs = list(first + s * np.arange(count))
            if qs[-1] > last + 1e-9:                      # last pass flush with the edge
                qs[-1] = last
        for q in qs:
            line = LineString([lo * u + q * n, hi * u + q * n])
            cut = poly.intersection(line)
            for piece in getattr(cut, "geoms", [cut]):
                if piece.is_empty or piece.length < 1e-6 or piece.geom_type != "LineString":
                    continue
                c = np.asarray(piece.coords)
                a, b = (c[0], c[-1]) if c[0] @ u < c[-1] @ u else (c[-1], c[0])
                if (b - a) @ u > 2 * inset:
                    a, b = a + inset * u, b - inset * u
                else:
                    a = b = (a + b) / 2
                out.append(Swath(a, b, float(q)))
    out.sort(key=lambda sw: (sw.q, float(sw.a @ u)))
    return out


def fastest_turn(mower: Mower, jump: float, reverse: bool = True,
                 smooth: bool = False) -> list[Segment]:
    """Fastest turn from the end of a pass to a pass ``jump`` to the side driven
    the other way (its rear axle 2 d further out, see turn_ahead); with
    ``smooth``, with curvature ramps (course, chapter 10)."""
    return list(_fastest_turn(mower, round(jump, 6), reverse, smooth))


@functools.lru_cache(maxsize=4096)
def _fastest_turn(mower: Mower, jump: float, reverse: bool, smooth: bool) -> tuple:
    a, b = Pose(0.0, 0.0, 0.0), Pose(2 * mower.deck_offset, jump, math.pi)
    if smooth:
        conn = connect(a, b, mower, None, reverse=reverse, smooth=True, smooth_words=16)
        if conn is not None:
            return tuple(conn.segments)
    from ..paths import fastest
    return tuple(fastest(a, b, mower, reverse=reverse))


def turn_time(mower: Mower, jump: float, reverse: bool = True, smooth: bool = False) -> float:
    """Time of the fastest turn over ``jump`` (see fastest_turn)."""
    p = fastest_turn(mower, jump, reverse, smooth)
    return path_time(p, mower.v_mow, mower.v_turn, mower.v_reverse, mower.t_shift)


def min_forward_jump(mower: Mower, spacing: float, smooth: bool = False) -> int:
    """Smallest number of passes j such that the fastest turn over j passes is
    driven forward only."""
    j = 1
    while any(sg.reverse for sg in fastest_turn(mower, j * spacing, smooth=smooth)):
        j += 1
    return j


def interleave(n: int, k: int) -> list[int]:
    """Blocks of 2k passes taken in the order i, i+k, i+1, i+k+1, ..."""
    order = []
    for i0 in range(0, n, 2 * k):
        for t in range(k):
            for j in (i0 + t, i0 + t + k):
                if j < n:
                    order.append(j)
    return order


def order_cost(order: list[int], cost: dict[int, float]) -> float:
    return sum(cost[abs(i - j)] for i, j in zip(order, order[1:]))


def improve_order(order: list[int], cost: dict[int, float]) -> list[int]:
    """Local search on the order of the passes: reverse a sub-sequence (2-opt) or
    move a run of one to three passes elsewhere (or-opt), while the total turn
    time decreases."""
    order = list(order)
    best = order_cost(order, cost)
    n = len(order)
    better = True
    while better:
        better = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                t = order_cost(new, cost)
                if t < best - 1e-9:
                    order, best, better = new, t, True
        for size in (1, 2, 3):
            for i in range(n - size + 1):
                run, rest = order[i:i + size], order[:i] + order[i + size:]
                for j in range(len(rest) + 1):
                    for r in (run, run[::-1]):
                        new = rest[:j] + r + rest[j:]
                        t = order_cost(new, cost)
                        if t < best - 1e-9:
                            order, best, better = new, t, True
    return order


def pass_order(n: int, pattern: str, mower: Mower | None = None,
               spacing: float = 0.92, smooth: bool = False) -> list[int]:
    """``adjacent``: 0, 1, 2, ...; ``skip``: the order of least total turn time,
    each turn timed by turn_time for its jump (local search from the adjacent
    order and from interleaved blocks)."""
    if pattern == "adjacent" or n <= 2:
        return list(range(n))
    cost = {j: turn_time(mower, j * spacing, smooth=smooth) for j in range(1, n)}
    starts = [list(range(n))] + [interleave(n, k) for k in range(2, 8)]
    return min((improve_order(o, cost) for o in starts), key=lambda o: order_cost(o, cost))


def width(region, theta: float) -> float:
    """Width of the region across the direction theta (extent along its normal)."""
    n = unit(theta + math.pi / 2)
    pts = np.vstack([np.asarray(p.exterior.coords) for p in polygons(region)])
    proj = pts @ n
    return float(proj.max() - proj.min())


def candidate_directions(region, step_deg: float = 15.0) -> list[float]:
    """Directions of the edges of the convex hull (where the width is smallest, see
    the course), plus a regular sweep."""
    hull = np.asarray(region.convex_hull.exterior.coords)
    dirs = {round(math.atan2(*(b - a)[::-1]) % math.pi, 6) for a, b in zip(hull, hull[1:])
            if np.linalg.norm(b - a) > 0.5}
    dirs |= {round(math.radians(d), 6) for d in np.arange(0, 180, step_deg)}
    return sorted(dirs)


def estimate_time(region, theta: float, mower: Mower, overlap: float, turn_time: float) -> float:
    """Quick estimate: pass lengths at mowing speed plus one turn per pass."""
    sw = swaths(region, theta, mower, overlap)
    length = sum(float(np.linalg.norm(x.b - x.a)) for x in sw)
    return length / mower.v_mow + turn_time * max(len(sw) - 1, 0)


def best_direction(region, mower: Mower, overlap: float = 0.05,
                   pattern: str = "adjacent") -> float:
    s = mower.cutting_width - overlap
    j = 1 if pattern == "adjacent" else min_forward_jump(mower, s)
    turn = turn_time(mower, j * s)
    return min(candidate_directions(region),
               key=lambda th: estimate_time(region, th, mower, overlap, turn))


@dataclass
class PassPlan:
    trajectory: Trajectory
    theta: float
    passes: int
    turns_backed_off: int
    turn_failures: int
    swaths: list[Swath]
    order: list[int]


def _pass_poses(x: Swath, theta: float, forward: bool, d: float):
    h = theta if forward else theta + math.pi
    e = unit(h)
    a, b = (x.a, x.b) if forward else (x.b, x.a)
    return a - d * e, b - d * e, h, e, float(np.linalg.norm(b - a))


def join(end: Pose, last_len: float, p_start, h: float, length: float, mower: Mower,
         checker: Checker, mow_reverse: bool = False, max_backoff: float = 3.0,
         step: float = 0.25, smooth: bool = False):
    """Fastest collision-free turn from the end of a pass to the start of the next.
    If none exists, the previous pass is stopped ``cp`` early and the next one
    started ``cn`` late, trying the pairs (cp, cn) by increasing cp + cn.
    Returns (connection, cp, cn) or None."""
    e = unit(h)
    grid = np.arange(0.0, max_backoff + 1e-9, step)
    pairs = sorted(((cp, cn) for cp in grid for cn in grid if cp <= last_len and cn <= length),
                   key=lambda c: (c[0] + c[1], c[0]))
    u = unit(end.theta)
    for cp, cn in pairs:
        s0 = Pose(end.x - cp * u[0], end.y - cp * u[1], end.theta)
        g = Pose(*(p_start + cn * e), h)
        conn = connect(s0, g, mower, checker, mow=True, mow_reverse=mow_reverse, smooth=smooth)
        if conn is not None:
            return conn, float(cp), float(cn)
    return None


def plan_passes(region, theta: float, mower: Mower, checker: Checker, start: Pose,
                pattern: str = "adjacent", overlap: float = 0.05, mow_reverse: bool = False,
                max_backoff: float = 3.0, step: float = 0.25,
                variant: tuple[bool, bool] | None = None, smooth: bool = False) -> PassPlan:
    """Mow the swaths of ``region`` in the given order, starting from ``start``.
    The order may be run backwards and the first pass driven either way: the
    variant (backwards, first pass forward) with the fastest transfer from
    ``start`` is kept, unless ``variant`` imposes one."""
    from .search import hybrid_astar
    s = mower.cutting_width - overlap
    sw = swaths(region, theta, mower, overlap)
    base = pass_order(len(sw), pattern, mower, s, smooth)
    d = mower.deck_offset
    variants = [variant] if variant is not None else [
        (rev, ff) for rev in (False, True) for ff in (True, False)]
    best = None
    for rev, first_forward in variants:
        order = base[::-1] if rev else base
        p0, _, h0, _, _ = _pass_poses(sw[order[0]], theta, first_forward, d)
        conn = connect(start, Pose(*p0, h0), mower, checker, mow=False, smooth=smooth)
        if conn is not None and (best is None or conn.time < best[0].time):
            best = (conn, order, first_forward)
    if best is None:                       # no direct path: search
        rev, first_forward = variants[0]
        order = base[::-1] if rev else base
        p0, _, h0, _, _ = _pass_poses(sw[order[0]], theta, first_forward, d)
        path = hybrid_astar(start, Pose(*p0, h0), mower, checker) or []
        best = (Connection(path, 0.0), order, first_forward)
    conn, order, first_forward = best
    traj = Trajectory(start).extend(conn.segments)
    backed = failures = 0
    prev = None                            # (end pose, length) of the previous pass
    for rank, j in enumerate(order):
        forward = (rank % 2 == 0) == first_forward
        p_start, p_end, h, e, length = _pass_poses(sw[j], theta, forward, d)
        cn = 0.0
        if prev is not None:
            res = join(prev[0], prev[1], p_start, h, length, mower, checker, mow_reverse,
                       max_backoff, step, smooth)
            if res is None:                # around an obstacle
                path = hybrid_astar(prev[0], Pose(*p_start, h), mower, checker)
                if path is None:
                    failures += 1
                    continue
                res = (Connection(path, 0.0), 0.0, 0.0)
            conn, cp, cn = res
            if cp > 0 or cn > 0:
                backed += 1
            if cp > 0:
                last = traj.segments[-1]
                traj.segments[-1] = Segment(0.0, last.length - cp, mow=True)
            traj.extend(conn.segments)
        traj.then(0.0, length - cn, mow=True)
        prev = (Pose(*p_end, h), length - cn)
    return PassPlan(traj, theta, len(sw), backed, failures, sw, order)


def plan_simple(lawn: Lawn, mower: Mower, pattern: str = "adjacent", theta: float | None = None,
                headland_passes: int | None = None, overlap: float = 0.05,
                margin: float = 0.05):
    """Headland, then parallel passes over the interior: a complete plan for a lawn
    whose interior every pass line crosses once (convex, or monotone in the chosen
    direction). Returns (trajectory, info)."""
    from .headland import headland_passes_needed, link_rings, plan_headland
    checker = Checker(lawn, mower, margin=margin)
    s = mower.cutting_width - overlap
    if headland_passes is None:
        if pattern == "skip":                     # forward turns over j* or j* + 1 passes
            j = min_forward_jump(mower, s)
            ahead = max(turn_ahead(mower, j * s), turn_ahead(mower, (j + 1) * s))
        else:
            ahead = turn_ahead(mower, s)
        c1 = 0.515 + margin + 0.05
        headland_passes = headland_passes_needed(mower, ahead, c1, s, margin)
    hl = plan_headland(lawn, mower, passes=headland_passes, overlap=overlap, margin=margin,
                       checker=checker)
    traj = link_rings(hl.rings, mower, checker)
    th = best_direction(hl.interior, mower, overlap, pattern) if theta is None else theta
    pp = plan_passes(hl.interior, th, mower, checker, traj.end, pattern, overlap)
    traj.extend(pp.trajectory.segments)
    info = dict(theta=th, headland_passes=headland_passes, passes=pp.passes,
                backed_off=pp.turns_backed_off, failures=pp.turn_failures,
                first_offset=hl.first_offset, headland=hl, passes_plan=pp)
    return traj, info


def turn_ahead(mower: Mower, jump: float, reverse: bool = True, smooth: bool = False) -> float:
    """How far the mower's outline goes beyond the rear axle at the end of a pass
    in the fastest turn to the pass ``jump`` to the side, driven the other way.
    The next pass starts with the deck at the same line as the end of this one,
    so its rear axle is 2 d further out (course, chapter 7)."""
    from .common import footprint_samples
    a = Pose(0.0, 0.0, 0.0)
    smp = Trajectory(a).extend(fastest_turn(mower, jump, reverse, smooth)).sample(0.02)
    pts = footprint_samples(mower, 0.05)
    return max(float((x + math.cos(t) * pts[:, 0] - math.sin(t) * pts[:, 1]).max())
               for x, t in zip(smp.x, smp.theta))
