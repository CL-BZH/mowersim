"""Lawns as polygons, in metres.

A lawn is an outer boundary with holes (flower beds, trees, a shed): the
mower must cut everything inside the boundary and outside the holes, and
its body must never leave that region.

Lawn files are JSON::

    {"name": "garden",
     "outer": [[x, y], ...],            # metres, any orientation
     "holes": [[[x, y], ...], ...],     # optional
     "start": [x, y, heading_deg]}      # optional: pose of the rear axle

Only numpy and matplotlib are used: the polygon area is the shoelace formula,
and the point-in-polygon test is matplotlib's (even-odd / crossing rule).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


def signed_area(poly: np.ndarray) -> float:
    """Shoelace formula: positive for a counter-clockwise polygon."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def orient(poly, ccw: bool) -> np.ndarray:
    """Return the polygon as an (n, 2) array, without a repeated last vertex, in the
    requested orientation."""
    p = np.asarray(poly, dtype=float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    if (signed_area(p) > 0) != ccw:
        p = p[::-1]
    return p


@dataclass(frozen=True)
class Pose:
    """Position of the reference point (middle of the rear axle) and heading in radians."""
    x: float
    y: float
    theta: float

    def __iter__(self):
        return iter((self.x, self.y, self.theta))


@dataclass(frozen=True)
class Lawn:
    outer: np.ndarray                       # (n, 2), counter-clockwise
    holes: tuple = ()                       # each (m, 2), clockwise
    name: str = "lawn"
    start: Pose | None = None
    _paths: tuple = field(default=(), repr=False, compare=False)

    @staticmethod
    def make(outer, holes=(), name="lawn", start=None) -> "Lawn":
        out = orient(outer, ccw=True)
        hs = tuple(orient(h, ccw=False) for h in holes)
        paths = (MplPath(out),) + tuple(MplPath(h) for h in hs)
        if start is not None and not isinstance(start, Pose):
            x, y, deg = start
            start = Pose(float(x), float(y), math.radians(deg))
        return Lawn(out, hs, name, start, paths)

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def from_dict(d: dict) -> "Lawn":
        return Lawn.make(d["outer"], d.get("holes", ()), d.get("name", "lawn"), d.get("start"))

    @staticmethod
    def from_file(path) -> "Lawn":
        d = json.loads(Path(path).read_text())
        d.setdefault("name", Path(path).stem)
        return Lawn.from_dict(d)

    def to_dict(self) -> dict:
        d = {"name": self.name, "outer": self.outer.round(4).tolist(),
             "holes": [h.round(4).tolist() for h in self.holes]}
        if self.start is not None:
            d["start"] = [self.start.x, self.start.y, round(math.degrees(self.start.theta), 4)]
        return d

    # ------------------------------------------------------------- geometry
    @property
    def area(self) -> float:
        """Area to mow: outer area minus the holes (holes are clockwise, so they add
        a negative signed area)."""
        return signed_area(self.outer) + sum(signed_area(h) for h in self.holes)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        (x0, y0), (x1, y1) = self.outer.min(axis=0), self.outer.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    def contains(self, pts) -> np.ndarray:
        """Boolean array: which points are inside the lawn (inside the outer boundary
        and outside every hole)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        inside = self._paths[0].contains_points(pts)
        for p in self._paths[1:]:
            inside &= ~p.contains_points(pts)
        return inside


# --------------------------------------------------------------- examples
def circle(cx: float, cy: float, r: float, n: int = 48) -> list:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)].round(4).tolist()


def rectangle(x0, y0, x1, y1) -> list:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def example_lawns() -> dict[str, Lawn]:
    """The lawns shipped in lawns/ (see tools/make_lawns.py)."""
    rect = Lawn.make(rectangle(0, 0, 20, 12), name="rectangle", start=(1.0, 1.0, 0))
    lshape = Lawn.make([[0, 0], [24, 0], [24, 9], [11, 9], [11, 20], [0, 20]],
                       name="l_shape", start=(1.0, 1.0, 0))
    # a garden with a curved border, a round tree and a rectangular flower bed
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    border = np.c_[17 + 15 * np.cos(t) + 1.5 * np.cos(3 * t), 11 + 9.5 * np.sin(t)
                   + 1.2 * np.sin(2 * t)]
    garden = Lawn.make(border.round(3).tolist(),
                       holes=[circle(10.5, 12.5, 1.2, 32), rectangle(20, 6, 25, 9)],
                       name="garden", start=(5.0, 9.0, 0))
    return {lawn.name: lawn for lawn in (rect, lshape, garden)}
