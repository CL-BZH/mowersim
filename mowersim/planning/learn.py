"""Estimating the time of a cell: a formula, a learned model, or a plan.

The planner chooses the direction of each cell (and the decomposition) with the
quick estimate of chapter 7: pass length at mowing speed plus one standard turn
per pass. That estimate ignores what makes some directions worse than others
on a real cell: turns against an oblique border, ends of passes that jump,
short passes. Chapter 12 compares three ways of choosing:

* the estimate (no planning);
* a model learned from planned cells: a linear regression on a few features of
  the swaths, fitted on random convex cells;
* search: plan the cell in each candidate direction and keep the fastest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from ..geometry import Lawn, Pose
from ..mower import Mower
from ..paths import path_time
from .cells import is_monotone
from .common import Checker, polygons
from .swaths import (_pass_poses, candidate_directions, estimate_time, plan_passes,
                     swaths, turn_time)

FEATURES = ("estimate (s)", "passes", "length of passes (m)", "end jumps (m)",
            "short passes", "oblique ends")


def random_cell(rng: np.random.Generator) -> Polygon:
    """A random convex cell: four to seven points on an ellipse of 6 to 20 m."""
    k = int(rng.integers(4, 8))
    a, b = rng.uniform(3, 10, 2)
    t = np.sort(rng.uniform(0, 2 * np.pi, k))
    rot = rng.uniform(0, np.pi)
    pts = np.c_[a * np.cos(t), b * np.sin(t)] @ np.array([[math.cos(rot), math.sin(rot)],
                                                          [-math.sin(rot), math.cos(rot)]])
    return Polygon(pts).convex_hull


def features(poly, theta: float, mower: Mower, overlap: float = 0.05) -> np.ndarray:
    """Features of a cell mowed in the direction theta (see FEATURES)."""
    s = mower.cutting_width - overlap
    sw = swaths(poly, theta, mower, overlap)
    lengths = np.array([float(np.linalg.norm(x.b - x.a)) for x in sw])
    u = np.array([math.cos(theta), math.sin(theta)])
    a = np.array([x.a @ u for x in sw])
    b = np.array([x.b @ u for x in sw])
    jumps = float(np.abs(np.diff(a)).sum() + np.abs(np.diff(b)).sum()) if len(sw) > 1 else 0.0
    # obliqueness: how far the ends move per pass, averaged (0 for a rectangle)
    oblique = jumps / max(len(sw) - 1, 1) / s
    est = estimate_time(poly, theta, mower, overlap, turn_time(mower, s))
    return np.array([est, len(sw), lengths.sum(), jumps, int((lengths < 2.0).sum()),
                     oblique])


def cell_time(poly, theta: float, mower: Mower, overlap: float = 0.05,
              pattern: str = "adjacent", checker: Checker | None = None,
              border: float = 2.02, smooth: bool = False) -> float | None:
    """Time to mow the cell with the planner of chapter 7, from the start of its
    first pass (no transfer). Unless a checker is given, the lawn is the cell
    enlarged by ``border``: the depth of two headland passes, as when the cell is
    the interior of a lawn."""
    sw = swaths(poly, theta, mower, overlap)
    if not sw:
        return None
    if checker is None:
        big = poly.buffer(border, join_style="mitre")
        lawn = Lawn.make(np.asarray(big.exterior.coords)[:-1])
        checker = Checker(lawn, mower, margin=0.05)
    p0, _, h0, _, _ = _pass_poses(sw[0], theta, True, mower.deck_offset)
    pp = plan_passes(poly, theta, mower, checker, Pose(*p0, h0), pattern, overlap,
                     variant=(False, True), smooth=smooth)
    if pp.turn_failures:
        return None
    m = mower
    return path_time(pp.trajectory.segments, m.v_mow, m.v_turn, m.v_reverse, m.t_shift)


@dataclass
class Model:
    """Linear model: time = w . [1, features]."""
    weights: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.c_[np.ones(len(x)), x] @ self.weights


def fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> Model:
    """Least squares with a little ridge regularisation (features scaled)."""
    a = np.c_[np.ones(len(x)), x]
    scale = np.maximum(np.abs(a).max(axis=0), 1e-9)
    an = a / scale
    w = np.linalg.solve(an.T @ an + ridge * np.eye(a.shape[1]), an.T @ y)
    return Model(w / scale)


def dataset(n: int, mower: Mower, seed: int = 0):
    """n random (cell, direction) pairs with their features and planned time."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(ys) < n:
        poly = random_cell(rng)
        dirs = candidate_directions(poly)
        theta = float(dirs[int(rng.integers(len(dirs)))])
        t = cell_time(poly, theta, mower)
        if t is None:
            continue
        xs.append(features(poly, theta, mower))
        ys.append(t)
    return np.array(xs), np.array(ys)


def choose_direction(poly, mower: Mower, how: str, model: Model | None = None,
                     checker: Checker | None = None) -> float:
    """The direction of a cell chosen by ``estimate``, ``model`` or ``search``
    (among the candidate directions in which the cell is monotone)."""
    good = [t for t in candidate_directions(poly) if is_monotone(poly, t)]
    s = mower.cutting_width - 0.05
    if how == "estimate":
        turn = turn_time(mower, s)
        return min(good, key=lambda t: estimate_time(poly, t, mower, 0.05, turn))
    if how == "model":
        return min(good, key=lambda t: float(model.predict(features(poly, t, mower))[0]))
    times = [(cell_time(poly, t, mower, checker=checker), t) for t in good]
    return min((x for x in times if x[0] is not None), key=lambda x: x[0])[1]


def region_cells(region) -> list:
    return polygons(region)


def regret_study(n_cells: int, model: Model, mower: Mower, seed: int = 100):
    """For random cells, the time of every candidate direction (planned), and the
    extra time of the direction chosen by the estimate and by the model over the
    best one. Returns a list of (best, by estimate, by model) times."""
    rng = np.random.default_rng(seed)
    s = mower.cutting_width - 0.05
    turn = turn_time(mower, s)
    out = []
    while len(out) < n_cells:
        poly = random_cell(rng)
        good = [t for t in candidate_directions(poly) if is_monotone(poly, t)]
        times = {t: cell_time(poly, t, mower) for t in good}
        times = {t: v for t, v in times.items() if v is not None}
        if len(times) < 2:
            continue
        by_est = min(times, key=lambda t: estimate_time(poly, t, mower, 0.05, turn))
        by_mod = min(times, key=lambda t: float(model.predict(features(poly, t, mower))[0]))
        out.append((min(times.values()), times[by_est], times[by_mod]))
    return out
