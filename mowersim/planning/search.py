"""Collision-free paths between two poses among obstacles: Hybrid A*.

The search (after Dolgov, Thrun, Montemerlo and Diebel, 2010) explores poses
reached by short arcs of a few fixed curvatures, forward and in reverse, and
keeps the best pose found in each cell of a grid over (x, y, theta). Its cost
is the driving time of the path model (path_time), with a penalty at each
change of direction. Two ideas make it fast:

* the heuristic is the larger of two lower bounds on the remaining time: the
  Reeds-Shepp length to the goal ignoring obstacles, and the obstacle-aware
  grid distance to the goal ignoring the turning radius, both divided by the
  highest speed;
* every few expansions it tries to finish with a direct Reeds-Shepp path
  (``connect``), which succeeds as soon as the goal is in sight.
"""
from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np

from ..geometry import Pose
from ..kinematics import Segment, advance
from ..mower import Mower
from ..paths import candidates, pify
from .common import Checker, connect


def grid_distance(checker: Checker, goal: Pose, res: float = 0.25) -> tuple:
    """Distance to the goal over the free cells of a coarse grid (8-connected BFS
    with Euclidean steps): a lower bound on the length of any path of the rear axle
    that ignores the turning radius."""
    r = checker.raster
    k = max(int(round(res / r.res)), 1)
    ny, nx = r.shape[0] // k, r.shape[1] // k
    free = r.mask[:ny * k, :nx * k].reshape(ny, k, nx, k).any(axis=(1, 3))
    dist = np.full((ny, nx), np.inf)
    gi = int((goal.y - r.y0) / (r.res * k))
    gj = int((goal.x - r.x0) / (r.res * k))
    if not (0 <= gi < ny and 0 <= gj < nx):
        return dist, k
    dist[gi, gj] = 0.0
    step = r.res * k
    # Dijkstra on the grid (small graph: a heap is fine)
    heap = [(0.0, gi, gj)]
    moves = [(di, dj, math.hypot(di, dj) * step) for di in (-1, 0, 1) for dj in (-1, 0, 1)
             if di or dj]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, c in moves:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and free[a, b] and d + c < dist[a, b]:
                dist[a, b] = d + c
                heapq.heappush(heap, (d + c, a, b))
    return dist, k


def hybrid_astar(start: Pose, goal: Pose, mower: Mower, checker: Checker,
                 step: float = 0.6, xy_res: float = 0.3, theta_bins: int = 72,
                 shot_every: int = 10, max_expansions: int = 6000,
                 trace: list | None = None) -> list[Segment] | None:
    """Segments (blades off) of a collision-free path from start to goal, or None.
    If ``trace`` is a list, the expanded poses are appended to it."""
    if not checker.free_poses(goal.x, goal.y, goal.theta)[0]:
        return None
    direct = connect(start, goal, mower, checker, mow=False)
    if direct is not None:
        return direct.segments
    k = mower.max_curvature
    curvatures = (-k, -k / 2, 0.0, k / 2, k)
    v_max = max(mower.v_mow, mower.v_turn, mower.v_reverse)
    dist, kk = grid_distance(checker, goal)
    r = checker.raster
    cell = r.res * kk

    def h(p: Pose) -> float:
        i = int((p.y - r.y0) / cell)
        j = int((p.x - r.x0) / cell)
        g2 = dist[i, j] if 0 <= i < dist.shape[0] and 0 <= j < dist.shape[1] else np.inf
        rs = candidates(p, goal, mower.min_radius)[0].length * mower.min_radius
        return max(rs, g2) / v_max

    def key(p: Pose) -> tuple:
        return (int(math.floor(p.x / xy_res)), int(math.floor(p.y / xy_res)),
                int(math.floor(pify(p.theta) / (2 * math.pi) * theta_bins)) % theta_bins)

    # node: (pose, cost, parent index, segment from parent, reverse flag)
    nodes = [(start, 0.0, -1, None, None)]
    best = {key(start): 0.0}
    heap = [(h(start), 0, 0)]
    counter = 1
    expansions = 0
    while heap and expansions < max_expansions:
        _, _, idx = heapq.heappop(heap)
        pose, cost, _, _, rev = nodes[idx]
        expansions += 1
        if trace is not None:
            trace.append(pose)
        if expansions % shot_every == 0:
            shot = connect(pose, goal, mower, checker, mow=False)
            if shot is not None:
                return _path(nodes, idx) + shot.segments
        for backwards in (False, True):
            for kappa in curvatures:
                seg = Segment(kappa, step, reverse=backwards, mow=False)
                nxt = advance(pose, kappa, seg.signed_length)
                v = mower.v_reverse if backwards else (mower.v_mow if kappa == 0
                                                       else mower.v_turn)
                c = cost + step / v + (mower.t_shift if rev is not None
                                       and rev != backwards else 0.0)
                kk_ = key(nxt)
                if c >= best.get(kk_, math.inf):
                    continue
                if not checker.free(pose, [seg]):
                    continue
                best[kk_] = c
                nodes.append((nxt, c, idx, seg, backwards))
                heapq.heappush(heap, (c + h(nxt), counter, len(nodes) - 1))
                counter += 1
    return None


def _path(nodes, idx: int) -> list[Segment]:
    segs = deque()
    while nodes[idx][2] >= 0:
        segs.appendleft(nodes[idx][3])
        idx = nodes[idx][2]
    # merge consecutive identical primitives into longer segments
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].curvature == s.curvature and out[-1].reverse == s.reverse:
            out[-1] = Segment(s.curvature, out[-1].length + s.length, s.reverse, s.mow)
        else:
            out.append(s)
    return out
