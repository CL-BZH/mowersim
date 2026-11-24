"""Shared tools of the planners: the lawn as a Shapely polygon, a fast collision
checker for the mower's footprint, and the search for a collision-free
Reeds-Shepp connection between two poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from ..coverage import Raster
from ..geometry import Lawn, Pose
from ..kinematics import Segment, Trajectory
from ..mower import Mower
from ..paths import candidates, path_time


def lawn_polygon(lawn: Lawn) -> Polygon:
    return orient(Polygon(lawn.outer, [h for h in lawn.holes]), sign=1.0)


def polygons(geom) -> list[Polygon]:
    """The polygons of a Shapely result (Polygon, MultiPolygon or empty)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def lateral_half_width(mower: Mower) -> float:
    return float(max(np.abs(mower.body_outline()[:, 1]).max(),
                     np.abs(mower.deck_outline()[:, 1]).max()))


def front_reach(mower: Mower) -> float:
    return float(max(mower.body_outline()[:, 0].max(), mower.deck_outline()[:, 0].max()))


def rear_reach(mower: Mower) -> float:
    return float(-min(mower.body_outline()[:, 0].min(), mower.deck_outline()[:, 0].min()))


def deck_front(mower: Mower) -> float:
    """How far ahead of the rear axle the blades cut."""
    return mower.deck_offset + mower.blade_radius


def footprint_samples(mower: Mower, res: float) -> np.ndarray:
    """Points on the outline of the body and deck, plus a grid inside them, in the
    mower frame: an obstacle cannot hide under the mower."""
    pts = [mower.footprint_points(res)]
    for poly in (mower.body_outline(), mower.deck_outline()):
        (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
        gx, gy = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1))
        pts.append(np.c_[gx.ravel(), gy.ravel()])
    return np.vstack(pts)


class Checker:
    """Is the mower's footprint on the lawn (shrunk by a safety margin)?"""

    def __init__(self, lawn: Lawn, mower: Mower, margin: float = 0.05, res: float = 0.05):
        self.mower = mower
        self.margin = margin
        poly = lawn_polygon(lawn).buffer(-margin, join_style="mitre") if margin > 0 \
            else lawn_polygon(lawn)
        parts = polygons(poly)
        # the raster of the shrunk lawn (holes grown by the margin)
        shrunk = Lawn.make(np.asarray(parts[0].exterior.coords)[:-1],
                           [np.asarray(i.coords)[:-1] for p in parts for i in p.interiors],
                           name=lawn.name)
        self.raster = Raster.of(shrunk, res, pad=2.0)
        # other parts of a lawn split by the margin: add them to the mask
        for p in parts[1:]:
            extra = Lawn.make(np.asarray(p.exterior.coords)[:-1],
                              [np.asarray(i.coords)[:-1] for i in p.interiors])
            r = self.raster
            ny, nx = r.shape
            xs = r.x0 + (np.arange(nx) + 0.5) * r.res
            ys = r.y0 + (np.arange(ny) + 0.5) * r.res
            gx, gy = np.meshgrid(xs, ys)
            r.mask |= extra.contains(np.c_[gx.ravel(), gy.ravel()]).reshape(ny, nx)
        self.pts = footprint_samples(mower, res)

    def free_poses(self, x, y, theta) -> np.ndarray:
        """Boolean array: for each pose, is the whole footprint on the lawn?"""
        x, y, theta = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x, y, theta))
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        px = x[:, None] + c * self.pts[None, :, 0] - s * self.pts[None, :, 1]
        py = y[:, None] + s * self.pts[None, :, 0] + c * self.pts[None, :, 1]
        r = self.raster
        col = np.floor((px - r.x0) / r.res).astype(np.int64)
        row = np.floor((py - r.y0) / r.res).astype(np.int64)
        ny, nx = r.shape
        inside = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        ok = np.zeros(px.shape, dtype=bool)
        ok[inside] = r.mask[row[inside], col[inside]]
        return ok.all(axis=1)

    def free(self, start: Pose, segments: list[Segment], step: float = 0.025) -> bool:
        if not segments:
            return bool(self.free_poses(start.x, start.y, start.theta)[0])
        smp = Trajectory(start, list(segments)).sample(step)
        # check in chunks, stopping at the first collision
        for i in range(0, len(smp.x), 400):
            if not self.free_poses(smp.x[i:i + 400], smp.y[i:i + 400],
                                   smp.theta[i:i + 400]).all():
                return False
        return True


@dataclass
class Connection:
    segments: list[Segment]
    time: float


def connect(start: Pose, goal: Pose, mower: Mower, checker: Checker | None,
            mow: bool = True, mow_reverse: bool = False, reverse: bool = True,
            max_words: int = 48) -> Connection | None:
    """Fastest Reeds-Shepp candidate word from start to goal whose footprint stays on
    the lawn (None if every candidate collides). Blades follow ``mow`` forward and
    ``mow_reverse`` in reverse."""
    best = None
    for w in candidates(start, goal, mower.min_radius, reverse)[:max_words]:
        segs = [Segment(s.curvature, s.length, s.reverse,
                        mow=(mow_reverse if s.reverse else mow))
                for s in w.segments(mower.min_radius)]
        t = path_time(segs, mower.v_mow, mower.v_turn, mower.v_reverse, mower.t_shift)
        if best is not None and t >= best.time:
            continue
        if checker is None or checker.free(start, segs):
            best = Connection(segs, t)
    return best


def unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])
