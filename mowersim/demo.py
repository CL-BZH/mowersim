"""A deliberately simple trajectory, to exercise the simulator: back and forth.

This is *not* a planner. It mows parallel passes across the bounding box of
the lawn and joins them with the fastest Reeds-Shepp turn. On a rectangle it
shows the two problems a real planner must solve:

* ``room_for_turns=False``: the passes run to the edge, and the turns at their
  ends drive the mower off the lawn (collisions);
* ``room_for_turns=True``: the passes stop early enough for the turns to stay
  on the lawn, and strips at both ends are left uncut.

The answer, headland passes around the border, is the subject of a later
chapter of the course.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Lawn, Pose
from .kinematics import Trajectory
from .mower import Mower, to_world
from .paths import fastest


def lateral_half_width(mower: Mower) -> float:
    """Half the width of the mower (body or deck housing, whichever is wider)."""
    return float(max(np.abs(mower.body_outline()[:, 1]).max(),
                     np.abs(mower.deck_outline()[:, 1]).max()))


def front_reach(mower: Mower) -> float:
    """How far the mower extends in front of the rear axle."""
    return float(max(mower.body_outline()[:, 0].max(), mower.deck_outline()[:, 0].max()))


def rear_reach(mower: Mower) -> float:
    return float(-min(mower.body_outline()[:, 0].min(), mower.deck_outline()[:, 0].min()))


def turn_extent(mower: Mower, spacing: float) -> tuple[float, float, float]:
    """Bounding box of the mower's outline during the turn from the end of a pass
    (rear axle at the origin, heading +x) to the next pass (y = spacing, heading -x):
    how far it goes beyond x = 0 forward, above y = spacing and below y = 0."""
    a, b = Pose(0.0, 0.0, 0.0), Pose(0.0, spacing, math.pi)
    traj = Trajectory(a).extend(fastest(a, b, mower))
    smp = traj.sample(0.02)
    pts = mower.footprint_points(0.05)
    world = np.vstack([to_world(pts, x, y, t) for x, y, t in zip(smp.x, smp.y, smp.theta)])
    return (float(world[:, 0].max()), float(world[:, 1].max() - spacing),
            float(-world[:, 1].min()))


def back_and_forth(lawn: Lawn, mower: Mower, overlap: float = 0.05,
                   room_for_turns: bool = True, mow_reverse: bool = False,
                   margin: float = 0.05) -> Trajectory:
    x0, y0, x1, y1 = lawn.bounds
    spacing = mower.cutting_width - overlap
    half = lateral_half_width(mower)
    low, high = y0 + half + margin, y1 - half - margin    # range of the pass centre lines
    # Every pass runs between the same two rear-axle positions, west_x and east_x,
    # where the mower turns. The turn sticks out ``ahead`` beyond the turning point in
    # the direction of the pass (the turn at the west end is the mirror image of the
    # turn at the east end); without room for turns, only the body must fit.
    if room_for_turns:
        ahead, above, below = turn_extent(mower, spacing)
        # the turn into the last pass bulges beyond it (and the first one below it)
        low = max(low, y0 + below + margin)
        high = min(high, y1 - above - margin)
    else:
        ahead = front_reach(mower)
    west_x, east_x = x0 + margin + ahead, x1 - margin - ahead
    if east_x <= west_x:
        raise ValueError("lawn too short for passes and turns")
    # evenly spaced passes: every turn is then the same (mirrored) turn, the one
    # whose extent was measured (a closer last pass would need a wider turn)
    ys = np.arange(low, high + 1e-9, spacing)
    if len(ys) == 0:
        raise ValueError("lawn too narrow for one pass")
    traj = Trajectory(Pose(west_x, float(ys[0]), 0.0))
    for i, y in enumerate(ys):
        east = i % 2 == 0
        if i > 0:
            goal = Pose(west_x if east else east_x, float(y), 0.0 if east else math.pi)
            for seg in fastest(traj.end, goal, mower):
                traj.then(seg.curvature, seg.length, seg.reverse,
                          mow=(not seg.reverse) or mow_reverse)
        traj.then(0.0, east_x - west_x, mow=True)
    return traj
