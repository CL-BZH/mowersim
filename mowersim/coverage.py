"""Measure a trajectory: area cut, missed and cut twice, collisions, time.

This is the single referee of the project: every planner returns a
``Trajectory`` and ``evaluate`` scores it, so planners cannot disagree on what
"mowed" means.

Coverage is measured on a raster of square cells of side ``res`` (5 cm by
default). The trajectory is sampled finely enough that no blade tip moves more
than ``res / 2`` between two samples, and at each sample where the blades are
engaged, every cell whose centre lies inside a blade circle is cut.

A cell cut, then left, then cut again counts as a second *pass*: the samples
where it is cut are grouped into passes, a new pass starting when more than
``gap`` metres of driving separate two cuts (by default the deck length plus a
margin, so one pass of the deck over a cell counts once).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Lawn
from .kinematics import Samples, Trajectory
from .mower import Mower, to_world
from .paths import path_time


@dataclass
class Raster:
    x0: float
    y0: float
    res: float
    mask: np.ndarray            # (ny, nx) True on lawn cells

    @staticmethod
    def of(lawn: Lawn, res: float = 0.05, pad: float = 0.0) -> "Raster":
        bx0, by0, bx1, by1 = lawn.bounds
        x0, y0 = bx0 - pad, by0 - pad
        nx = int(math.ceil((bx1 + pad - x0) / res))
        ny = int(math.ceil((by1 + pad - y0) / res))
        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y0 + (np.arange(ny) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)
        mask = lawn.contains(np.c_[gx.ravel(), gy.ravel()]).reshape(ny, nx)
        return Raster(x0, y0, res, mask)

    @property
    def shape(self):
        return self.mask.shape

    def cell_of(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of the cell containing each point (may be out of range)."""
        col = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
        row = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
        return row, col

    def on_lawn(self, pts: np.ndarray) -> np.ndarray:
        row, col = self.cell_of(pts)
        ny, nx = self.shape
        ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        out = np.zeros(len(pts), dtype=bool)
        out[ok] = self.mask[row[ok], col[ok]]
        return out


def effective_width(mower: Mower, curvature: float) -> float:
    """Area swept by the blades per metre driven by the rear-axle point, on an arc of
    the given curvature (Proposition "swept area per metre" of the course).

    Straight: the cutting width w. On an arc of radius R about the instantaneous
    centre of rotation C, the blade circles sweep the annulus between
    rho_in = min_i(|c_i - C| - r) (or 0) and rho_out = max_i(|c_i - C| + r), so
    per radian the area is (rho_out^2 - rho_in^2) / 2, and per metre of the rear
    axle (R metres per radian) it is (rho_out^2 - rho_in^2) / (2 R).

    This is an upper bound on the area newly cut per metre; it is reached when
    the blade circles leave no gap between them.
    """
    if abs(curvature) < 1e-12:
        return mower.cutting_width
    radius = 1.0 / abs(curvature)
    centre = np.array([0.0, radius])                     # left turn; right is symmetric
    c = mower.blade_centres.copy()
    if curvature < 0:
        c[:, 1] = -c[:, 1]
    d = np.linalg.norm(c - centre, axis=1)
    rho_out = float(np.max(d + mower.blade_radius))
    rho_in = max(0.0, float(np.min(d - mower.blade_radius)))
    return (rho_out ** 2 - rho_in ** 2) / (2 * radius)


def max_effective_width(mower: Mower) -> float:
    ks = np.linspace(-mower.max_curvature, mower.max_curvature, 401)
    return max(effective_width(mower, k) for k in ks)


def blade_area(mower: Mower) -> float:
    """Area under the blades at standstill (upper bound: the sum of the discs)."""
    return len(mower.blade_centres) * math.pi * mower.blade_radius ** 2


def length_lower_bound(lawn: Lawn, mower: Mower, engagements: int = 1,
                       area: float | None = None) -> float:
    """No trajectory can cut ``area`` (default: the whole lawn) with less mowing distance
    than (area - n a0) / w_max, where n is the number of times the blades are engaged,
    a0 the area under the blades when they are engaged, and w_max the largest
    effective width over all allowed curvatures (Theorem "lower bound" of the course)."""
    a = lawn.area if area is None else area
    return max(a - engagements * blade_area(mower), 0.0) / max_effective_width(mower)


@dataclass
class Result:
    lawn_area: float            # m^2
    cut_area: float             # lawn area cut at least once
    missed_area: float
    overlap_area: float         # lawn area cut in two passes or more
    mean_passes: float          # over the cut lawn area
    length: float               # rear-axle distance, forward + reverse (m)
    length_reverse: float
    length_mowing: float        # with the blades engaged
    direction_changes: int
    time: float                 # s
    lower_bound: float          # minimum mowing length for the whole lawn, blades engaged
                                # once (m), see length_lower_bound
    engagements: int            # times the blades are switched on
    collisions: int             # samples where part of the mower is off the lawn
    first_collision: tuple | None
    max_curvature_ratio: float  # max |kappa| / kappa_max (> 1: not drivable)
    passes: np.ndarray          # (ny, nx) passes per cell
    raster: Raster

    @property
    def coverage(self) -> float:
        return self.cut_area / self.lawn_area if self.lawn_area else 1.0

    efficiency: float = 0.0     # minimum mowing length to cut the area actually cut (with
                                # its engagements) / actual mowing length; at most 1

    @property
    def feasible(self) -> bool:
        return self.collisions == 0 and self.max_curvature_ratio <= 1 + 1e-9

    def as_dict(self) -> dict:
        return {
            "coverage_%": round(100 * self.coverage, 2),
            "missed_m2": round(self.missed_area, 2),
            "overlap_m2": round(self.overlap_area, 2),
            "length_m": round(self.length, 1),
            "reverse_m": round(self.length_reverse, 1),
            "direction_changes": self.direction_changes,
            "time_min": round(self.time / 60, 2),
            "lower_bound_m": round(float(self.lower_bound), 1),
            "efficiency": round(float(self.efficiency), 3),
            "collisions": self.collisions,
        }


def sample_step(mower: Mower, res: float) -> float:
    """Rear-axle step such that no blade tip moves more than res/2 between samples.

    A point at distance rho from the centre of rotation moves rho/R times as fast as
    the rear axle; the farthest blade tip is at most at R + |deck| from it."""
    reach = math.hypot(mower.deck_offset, max(abs(y) for y in mower.blade_lateral)) \
        + mower.blade_radius
    factor = (mower.min_radius + reach) / mower.min_radius
    return 0.5 * res / factor


def cut_cells(samples: Samples, mower: Mower, raster: Raster, gap: float | None = None):
    """Passes per cell: (ny, nx) integer array."""
    res = raster.res
    ny, nx = raster.shape
    passes = np.zeros(ny * nx, dtype=np.int32)
    on = np.flatnonzero(samples.mow)
    if len(on) == 0:
        return passes.reshape(ny, nx)
    if gap is None:
        gap = 2 * (mower.blade_radius + mower.deck_margin) + 0.1
    r = mower.blade_radius
    # stencil: offsets of the cells that may intersect a blade circle
    k = int(math.ceil(r / res)) + 1
    oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
    oy, ox = oy.ravel(), ox.ravel()
    cells, when = [], []
    chunk = 4000
    for c in mower.blade_centres:
        for i in range(0, len(on), chunk):
            idx = on[i:i + chunk]
            cos, sin = np.cos(samples.theta[idx]), np.sin(samples.theta[idx])
            cx = samples.x[idx] + cos * c[0] - sin * c[1]        # blade centre, world frame
            cy = samples.y[idx] + sin * c[0] + cos * c[1]
            row0 = np.floor((cy - raster.y0) / res).astype(np.int64)
            col0 = np.floor((cx - raster.x0) / res).astype(np.int64)
            rows = row0[:, None] + oy[None, :]
            cols = col0[:, None] + ox[None, :]
            # distance from the blade centre to each cell centre
            dx = raster.x0 + (cols + 0.5) * res - cx[:, None]
            dy = raster.y0 + (rows + 0.5) * res - cy[:, None]
            hit = (dx * dx + dy * dy <= r * r) & (rows >= 0) & (rows < ny) & (cols >= 0) \
                & (cols < nx)
            cells.append((rows * nx + cols)[hit])
            when.append(np.broadcast_to(samples.s[idx][:, None], hit.shape)[hit])
    cells = np.concatenate(cells)
    when = np.concatenate(when)
    order = np.lexsort((when, cells))
    cells, when = cells[order], when[order]
    first = np.ones(len(cells), dtype=bool)
    first[1:] = (cells[1:] != cells[:-1]) | (when[1:] - when[:-1] > gap)
    np.add.at(passes, cells[first], 1)
    return passes.reshape(ny, nx)


def collisions(samples: Samples, mower: Mower, raster: Raster, every: int = 1):
    """Samples where a point of the body or deck outline is off the lawn."""
    pts = mower.footprint_points(spacing=raster.res)
    bad = []
    for i in range(0, len(samples.x), every):
        w = to_world(pts, samples.x[i], samples.y[i], samples.theta[i])
        if not raster.on_lawn(w).all():
            bad.append(i)
    return bad


def evaluate(lawn: Lawn, mower: Mower, traj: Trajectory, res: float = 0.05,
             check_collisions: bool = True) -> Result:
    raster = Raster.of(lawn, res, pad=1.0)
    samples = traj.sample(sample_step(mower, res))
    passes = cut_cells(samples, mower, raster)
    lawn_cells = raster.mask
    cell_area = res * res
    cut = (passes > 0) & lawn_cells
    bad = []
    if check_collisions:
        # check about every 2.5 cm of travel: enough for a 5 cm raster
        every = max(int(round(0.025 / (samples.s[1] - samples.s[0]))) if len(samples.s) > 1
                    else 1, 1)
        bad = collisions(samples, mower, raster, every)
    segs = traj.segments
    # the blades are engaged at the start of every run of segments with mow=True
    engagements = sum(1 for i, sg in enumerate(segs)
                      if sg.mow and (i == 0 or not segs[i - 1].mow))
    traj_mow_length = sum(sg.length for sg in segs if sg.mow)
    cut_area = float(cut.sum() * cell_area)
    t = path_time(segs, mower.v_mow, mower.v_turn, mower.v_reverse, mower.t_shift)
    return Result(
        lawn_area=float(lawn_cells.sum() * cell_area),
        cut_area=cut_area,
        missed_area=float((lawn_cells & ~cut).sum() * cell_area),
        overlap_area=float(((passes > 1) & lawn_cells).sum() * cell_area),
        mean_passes=float(passes[cut].mean()) if cut.any() else 0.0,
        length=traj.length,
        length_reverse=sum(s.length for s in segs if s.reverse),
        length_mowing=sum(s.length for s in segs if s.mow),
        direction_changes=traj.direction_changes(),
        time=t,
        lower_bound=length_lower_bound(lawn, mower),
        engagements=engagements,
        efficiency=(length_lower_bound(lawn, mower, engagements, cut_area)
                    / traj_mow_length if traj_mow_length else 0.0),
        collisions=len(bad),
        first_collision=(float(samples.x[bad[0]]), float(samples.y[bad[0]])) if bad else None,
        max_curvature_ratio=traj.max_curvature() / mower.max_curvature,
        passes=passes,
        raster=raster,
    )
