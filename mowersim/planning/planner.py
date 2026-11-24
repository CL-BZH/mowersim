"""The complete planner: headland, cells, and the order of the cells.

Each cell can be mowed in four ways (its order of passes forwards or
backwards, the first pass driven either way), each with its own entry and exit
pose. Choosing the order of the cells and the way each is mowed is a
generalised travelling salesman problem (one "city" to visit per cell, among
four poses); with a handful of cells it is solved exactly by dynamic
programming over the subsets of cells (Held and Karp). The cost of a transfer
is estimated without searching (course, chapter 9); the transfers are then
planned: the fastest collision-free Reeds-Shepp path, or Hybrid A*.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..geometry import Lawn, Pose
from ..kinematics import Trajectory
from ..mower import Mower
from ..paths import fastest, path_time
from .cells import Cell, best_decomposition
from .common import Checker, connect
from .search import grid_distance
from .swaths import _pass_poses, min_forward_jump, pass_order, plan_passes, swaths, turn_ahead

VARIANTS = [(rev, ff) for rev in (False, True) for ff in (True, False)]


@dataclass
class Way:
    """One of the four ways of mowing a cell."""
    variant: tuple[bool, bool]       # (order backwards, first pass forward)
    entry: Pose                      # rear axle at the start of the first pass
    exit: Pose                       # rear axle at the end of the last pass


def cell_ways(cell: Cell, mower: Mower, pattern: str, overlap: float) -> list[Way]:
    s = mower.cutting_width - overlap
    sw = swaths(cell.polygon, cell.theta, mower, overlap)
    if not sw:
        return []
    base = pass_order(len(sw), pattern, mower, s)
    d = mower.deck_offset
    out = []
    for rev, ff in VARIANTS:
        order = base[::-1] if rev else base
        p0, _, h0, _, _ = _pass_poses(sw[order[0]], cell.theta, ff, d)
        last_forward = ((len(order) - 1) % 2 == 0) == ff
        _, p1, h1, _, _ = _pass_poses(sw[order[-1]], cell.theta, last_forward, d)
        out.append(Way((rev, ff), Pose(*p0, h0), Pose(*p1, h1)))
    return out


class TransferCost:
    """Estimated time of a transfer between two poses, blades off: the larger of
    the fastest Reeds-Shepp time (ignoring obstacles) and the grid distance
    around the obstacles at mowing speed (ignoring the turning radius). Both are
    optimistic; their maximum is the better estimate. With ``direct``, the time
    of the fastest collision-free Reeds-Shepp path is used when there is one."""

    def __init__(self, mower: Mower, checker: Checker, direct: bool = True):
        self.mower, self.checker, self.direct = mower, checker, direct
        self._grids: dict = {}

    def __call__(self, a: Pose, b: Pose) -> float:
        m = self.mower
        if self.direct:                  # a collision-free Reeds-Shepp path: exact
            conn = connect(a, b, m, self.checker, mow=False)
            if conn is not None:
                return conn.time
        rs = path_time(fastest(a, b, m), m.v_mow, m.v_turn, m.v_reverse, m.t_shift)
        key = (round(b.x, 3), round(b.y, 3))
        if key not in self._grids:
            self._grids[key] = grid_distance(self.checker, b)
        dist, k = self._grids[key]
        r = self.checker.raster
        cell = r.res * k
        i, j = int((a.y - r.y0) / cell), int((a.x - r.x0) / cell)
        g = dist[i, j] if 0 <= i < dist.shape[0] and 0 <= j < dist.shape[1] else math.inf
        return max(rs, g / m.v_mow if np.isfinite(g) else rs)


def order_cells(start: Pose, ways: list[list[Way]], cost) -> tuple[list[tuple[int, int]], float]:
    """Exact order of the cells and way of mowing each (Held-Karp dynamic
    programming over subsets): the sequence of (cell, way) that minimises the
    total estimated transfer time from ``start``."""
    n = len(ways)
    if n == 0:
        return [], 0.0
    c_start = [[cost(start, w.entry) for w in ways[i]] for i in range(n)]
    c = {}
    for i, j in itertools.permutations(range(n), 2):
        for a, wa in enumerate(ways[i]):
            for b, wb in enumerate(ways[j]):
                c[i, a, j, b] = cost(wa.exit, wb.entry)
    best = {}                                    # (mask, i, a) -> (cost, previous)
    for i in range(n):
        for a in range(len(ways[i])):
            best[1 << i, i, a] = (c_start[i][a], None)
    for mask in range(1, 1 << n):
        for i in range(n):
            if not mask >> i & 1:
                continue
            for a in range(len(ways[i])):
                if (mask, i, a) not in best:
                    continue
                base = best[mask, i, a][0]
                for j in range(n):
                    if mask >> j & 1:
                        continue
                    for b in range(len(ways[j])):
                        key = (mask | 1 << j, j, b)
                        t = base + c[i, a, j, b]
                        if t < best.get(key, (math.inf,))[0]:
                            best[key] = (t, (mask, i, a))
    full = (1 << n) - 1
    end = min((k for k in best if k[0] == full), key=lambda k: best[k][0])
    total = best[end][0]
    seq = []
    k = end
    while k is not None:
        seq.append((k[1], k[2]))
        k = best[k][1]
    return seq[::-1], total


def greedy_order(start: Pose, ways: list[list[Way]], cost) -> tuple[list[tuple[int, int]], float]:
    """Each time the nearest (cell, way) not yet mowed."""
    todo, here, seq, total = set(range(len(ways))), start, [], 0.0
    while todo:
        t, i, a = min((cost(here, w.entry), i, a) for i in todo for a, w in enumerate(ways[i]))
        seq.append((i, a))
        total += t
        here = ways[i][a].exit
        todo.remove(i)
    return seq, total


def plan(lawn: Lawn, mower: Mower, pattern: str = "best", order: str = "optimal",
         overlap: float = 0.05, margin: float = 0.05, headland_passes: int | None = None):
    """Complete plan: headland, cells, cells in the best order. ``pattern`` is
    ``adjacent``, ``skip`` or ``best`` (both, keep the faster plan); ``order`` is
    ``optimal`` (dynamic programming) or ``greedy``. Returns (trajectory, info)."""
    if pattern == "best":
        from ..coverage import evaluate
        plans = [plan(lawn, mower, p, order, overlap, margin, headland_passes)
                 for p in ("adjacent", "skip")]
        timed = [(evaluate(lawn, mower, t).time, t, info) for t, info in plans]
        _, traj, info = min(timed, key=lambda x: x[0])
        info["compared"] = {p: round(t / 60, 2) for p, (t, _, _) in zip(("adjacent", "skip"),
                                                                          timed)}
        return traj, info
    from .headland import headland_passes_needed, link_rings, plan_headland
    checker = Checker(lawn, mower, margin=margin)
    s = mower.cutting_width - overlap
    if headland_passes is None:
        j = 1 if pattern == "adjacent" else min_forward_jump(mower, s)
        ahead = max(turn_ahead(mower, j * s), turn_ahead(mower, (j + 1) * s))
        headland_passes = headland_passes_needed(mower, ahead, 0.515 + margin + 0.05, s, margin)
    hl = plan_headland(lawn, mower, passes=headland_passes, overlap=overlap, margin=margin,
                       checker=checker)
    traj = link_rings(hl.rings, mower, checker)
    cells = [c for c in best_decomposition(hl.interior, mower, overlap, pattern)
             if swaths(c.polygon, c.theta, mower, overlap)]
    ways = [cell_ways(c, mower, pattern, overlap) for c in cells]
    cost = TransferCost(mower, checker)
    solve = order_cells if order == "optimal" else greedy_order
    seq, estimate = solve(traj.end, ways, cost)
    plans = []
    for i, a in seq:
        c = cells[i]
        pp = plan_passes(c.polygon, c.theta, mower, checker, traj.end, pattern, overlap,
                         variant=ways[i][a].variant)
        traj.extend(pp.trajectory.segments)
        plans.append((c, pp))
    info = dict(pattern=pattern, headland_passes=headland_passes, cells=cells, headland=hl,
                plans=plans, sequence=seq, transfer_estimate=estimate,
                failures=sum(pp.turn_failures for _, pp in plans),
                passes=sum(pp.passes for _, pp in plans))
    return traj, info


def transfer_segments(traj: Trajectory) -> float:
    """Distance driven with the blades off."""
    return float(sum(sg.length for sg in traj.segments if not sg.mow))


def time_breakdown(traj: Trajectory, mower: Mower) -> dict[str, float]:
    """Where the time goes (seconds): straight passes with the blades on, turns
    with the blades on (arcs and reverse), transfers with the blades off, and the
    stops to change direction."""
    m = mower
    out = dict(passes=0.0, turns=0.0, transfers=0.0, stops=0.0)
    prev = None
    for sg in traj.segments:
        if sg.length <= 0:
            continue
        v = m.v_reverse if sg.reverse else (m.v_mow if sg.curvature == 0 else m.v_turn)
        t = sg.length / v
        if not sg.mow:
            out["transfers"] += t
        elif sg.curvature == 0 and not sg.reverse:
            out["passes"] += t
        else:
            out["turns"] += t
        if prev is not None and prev.reverse != sg.reverse:
            out["stops"] += m.t_shift
        prev = sg
    return out
