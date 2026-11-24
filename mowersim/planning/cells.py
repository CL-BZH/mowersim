"""Cutting a region into cells that parallel passes can mow one after the other.

A region is *monotone* in the direction theta when every line parallel to theta
meets it in at most one segment: its passes then all run from one side of the
region to the other, and chapter 7's planner applies. The boustrophedon
decomposition (Choset, 2000) cuts any polygon with holes into monotone cells:

1. turn the region so that theta points along x;
2. cut it by the horizontal lines through all its vertices: every piece of a
   strip between two consecutive lines is monotone;
3. merge two pieces of consecutive strips when each is the only neighbour of
   the other across the line between them.

The cuts that survive step 3 are the ones through the *critical* vertices,
where the number of segments on a line changes (course, chapter 8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from ..mower import Mower
from .common import polygons
from .swaths import candidate_directions, estimate_time, min_forward_jump, swaths, turn_time

EPS = 1e-7


@dataclass
class Cell:
    polygon: object          # Shapely polygon
    theta: float             # direction of its passes
    index: int = 0


def _rotate(geom, theta: float, back: bool = False):
    return affinity.rotate(geom, theta if back else -theta, origin=(0, 0), use_radians=True)


def slices(region, theta: float) -> list[list]:
    """Step 1 and 2: the pieces of each strip, bottom to top, in the frame where
    theta points along x."""
    geom = _rotate(region, theta)
    ys = sorted({round(y, 9) for p in polygons(geom)
                 for ring in (p.exterior, *p.interiors) for _, y in ring.coords})
    x0, _, x1, _ = geom.bounds
    strips = []
    for ya, yb in zip(ys, ys[1:]):
        if yb - ya < 1e-6:
            continue
        cut = geom.intersection(box(x0 - 1, ya, x1 + 1, yb))
        strips.append([p for p in polygons(cut) if p.area > 1e-9])
    return strips


def _touch(a, b) -> float:
    """Length of the boundary that two pieces of consecutive strips share."""
    g = a.boundary.intersection(b.boundary)
    return getattr(g, "length", 0.0)


def _extent(piece, y: float) -> tuple[float, float]:
    """x-range of a piece of a strip along its bottom or top line y."""
    x0, _, x1, _ = piece.bounds
    g = piece.intersection(box(x0 - 1, y - 1e-4, x1 + 1, y + 1e-4))
    b = g.bounds
    return (b[0], b[2]) if b else (math.nan, math.nan)


def decompose(region, theta: float, min_area: float = 0.0,
              max_jump: float | None = 1.0) -> list[Cell]:
    """Boustrophedon decomposition of ``region`` into cells monotone in the
    direction theta. Two pieces are not merged either where the ends of the
    passes would jump by more than ``max_jump`` from one side of the line to
    the other (None: merge whenever one to one). Cells smaller than
    ``min_area`` are merged into the neighbour they share the longest boundary
    with, if the union stays monotone."""
    strips = slices(region, theta)
    cells: list = []                      # merged pieces (rotated frame)
    last: list = []                       # the last piece added to each cell
    top: list[int] = []                   # cells open at the top of the last strip
    for strip in strips:
        below = [(i, cells[i]) for i in top]
        nb_up = {i: [] for i, _ in below}
        nb_down = []
        for k, piece in enumerate(strip):
            nb = [i for i, c in below if _touch(c, piece) > 1e-6]
            nb_down.append(nb)
            for i in nb:
                nb_up[i].append(k)
        new_top = []
        for k, piece in enumerate(strip):
            nb = nb_down[k]
            merge = len(nb) == 1 and len(nb_up[nb[0]]) == 1      # one to one
            if merge and max_jump is not None:
                y = piece.bounds[1]
                a, b = _extent(last[nb[0]], y), _extent(piece, y)
                merge = abs(a[0] - b[0]) <= max_jump and abs(a[1] - b[1]) <= max_jump
            if merge:                                            # extend the cell
                i = nb[0]
                cells[i] = unary_union([cells[i], piece])
                last[i] = piece
                new_top.append(i)
            else:                                                # a new cell starts
                cells.append(piece)
                last.append(piece)
                new_top.append(len(cells) - 1)
        top = new_top
    out = [Cell(_rotate(c, theta, back=True), theta, i) for i, c in enumerate(cells)]
    if min_area > 0:
        out = merge_small(out, min_area)
    for i, c in enumerate(out):
        c.index = i
    return out


def is_monotone(poly, theta: float, step: float = 0.2) -> bool:
    """Every line parallel to theta, ``step`` apart, meets the polygon at most once."""
    g = _rotate(poly, theta)
    x0, y0, x1, y1 = g.bounds
    for y in np.arange(y0 + step / 2, y1, step):
        cut = g.intersection(LineString([(x0 - 1, y), (x1 + 1, y)]))
        if len(getattr(cut, "geoms", [cut])) > 1:
            return False
    return True


def merge_small(cells: list[Cell], min_area: float) -> list[Cell]:
    cells = list(cells)
    changed = True
    while changed:
        changed = False
        for c in sorted(cells, key=lambda c: c.polygon.area):
            if c.polygon.area >= min_area:
                break
            others = [o for o in cells if o is not c]
            shared = sorted(others, key=lambda o: -_touch(o.polygon, c.polygon))
            for o in shared:
                if _touch(o.polygon, c.polygon) < 1e-6:
                    break
                u = unary_union([o.polygon, c.polygon])
                if u.geom_type == "Polygon" and is_monotone(u, o.theta):
                    o.polygon = u
                    cells.remove(c)
                    changed = True
                    break
            if changed:
                break
    return cells


def cell_time(poly, theta: float, mower: Mower, overlap: float, turn: float) -> float:
    return estimate_time(poly, theta, mower, overlap, turn)


def best_cell_direction(poly, mower: Mower, overlap: float = 0.05,
                        pattern: str = "adjacent") -> float:
    """Among the candidate directions in which the cell is monotone, the one with
    the smallest estimated time (chapter 7)."""
    s = mower.cutting_width - overlap
    j = 1 if pattern == "adjacent" else min_forward_jump(mower, s)
    turn = turn_time(mower, j * s)
    good = [t for t in candidate_directions(poly) if is_monotone(poly, t)]
    return min(good, key=lambda t: estimate_time(poly, t, mower, overlap, turn))


def decomposition_time(cells: list[Cell], mower: Mower, overlap: float, turn: float,
                       per_cell: float) -> float:
    return sum(estimate_time(c.polygon, c.theta, mower, overlap, turn) for c in cells) \
        + per_cell * len(cells)


def best_decomposition(region, mower: Mower, overlap: float = 0.05, pattern: str = "adjacent",
                       per_cell: float = 30.0, min_area: float | None = None,
                       own_directions: bool = True) -> list[Cell]:
    """Try the decomposition in each candidate direction; in each cell, optionally
    pick the cell's own best direction; keep the decomposition with the smallest
    estimated time, counting ``per_cell`` seconds for reaching each cell."""
    s = mower.cutting_width - overlap
    j = 1 if pattern == "adjacent" else min_forward_jump(mower, s)
    turn = turn_time(mower, j * s)
    min_area = 4 * mower.cutting_width ** 2 if min_area is None else min_area
    best, best_t = None, math.inf
    for theta in candidate_directions(region):
        cells = decompose(region, theta, min_area)
        if own_directions:
            for c in cells:
                c.theta = best_cell_direction(c.polygon, mower, overlap, pattern)
        t = decomposition_time(cells, mower, overlap, turn, per_cell)
        if t < best_t:
            best, best_t = cells, t
    return best


def passes_per_line(poly, theta: float, mower: Mower, overlap: float = 0.05) -> int:
    """Largest number of separate swaths on one line (1 for a monotone cell)."""
    sw = swaths(poly, theta, mower, overlap)
    counts: dict[float, int] = {}
    for x in sw:
        counts[round(x.q, 6)] = counts.get(round(x.q, 6), 0) + 1
    return max(counts.values(), default=0)


def plan_cells(lawn, mower: Mower, pattern: str = "adjacent", overlap: float = 0.05,
               margin: float = 0.05, headland_passes: int | None = None,
               own_directions: bool = True):
    """Headland, decomposition of the interior, then the cells one after the other,
    each time the cell whose first pass is closest (a greedy order; chapter 9
    does better). Returns (trajectory, info)."""
    from .common import Checker
    from .headland import headland_passes_needed, link_rings, plan_headland
    from .swaths import plan_passes, turn_ahead
    checker = Checker(lawn, mower, margin=margin)
    s = mower.cutting_width - overlap
    if headland_passes is None:
        j = 1 if pattern == "adjacent" else min_forward_jump(mower, s)
        ahead = max(turn_ahead(mower, j * s), turn_ahead(mower, (j + 1) * s))
        headland_passes = headland_passes_needed(mower, ahead, 0.515 + margin + 0.05, s, margin)
    hl = plan_headland(lawn, mower, passes=headland_passes, overlap=overlap, margin=margin,
                       checker=checker)
    traj = link_rings(hl.rings, mower, checker)
    cells = best_decomposition(hl.interior, mower, overlap, pattern,
                               own_directions=own_directions)
    todo = list(cells)
    plans = []
    while todo:
        here = traj.end
        p = np.array([here.x, here.y])

        def gap(c):
            sw = swaths(c.polygon, c.theta, mower, overlap)
            if not sw:
                return math.inf
            ends = [sw[0].a, sw[0].b, sw[-1].a, sw[-1].b]
            return min(float(np.linalg.norm(e - p)) for e in ends)
        c = min(todo, key=gap)
        todo.remove(c)
        if not swaths(c.polygon, c.theta, mower, overlap):
            continue
        pp = plan_passes(c.polygon, c.theta, mower, checker, here, pattern, overlap)
        traj.extend(pp.trajectory.segments)
        plans.append((c, pp))
    info = dict(headland_passes=headland_passes, cells=cells, headland=hl, plans=plans,
                failures=sum(pp.turn_failures for _, pp in plans),
                passes=sum(pp.passes for _, pp in plans))
    return traj, info
