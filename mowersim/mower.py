"""The mower seen from above: dimensions, cutting deck and speeds.

Reference frame of the mower: origin at the middle of the rear axle, x forward,
y to the left. The middle of the rear axle is the point whose motion the
kinematic model describes (see kinematics.py): it moves along the heading, and
the curvature of its path is set by the steering angle.

The default values are those of a typical tractor-style ride-on mower with a
deck of about one metre and two blades. They are only defaults: load your own
from a JSON file with ``Mower.from_file``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mower:
    # chassis
    wheelbase: float = 1.20          # rear axle to front axle (m)
    track: float = 0.80              # distance between left and right wheels (m)
    max_steer_deg: float = 40.0      # largest angle of the (virtual) central front wheel
    rear_overhang: float = 0.45      # body behind the rear axle (m)
    front_overhang: float = 0.35     # body in front of the front axle (m)
    body_width: float = 0.95         # chassis including the wheels (m)
    wheel_diameter: float = 0.40
    wheel_width: float = 0.15
    # cutting deck: blade centres in the mower frame, all at x = deck_offset
    deck_offset: float = 0.55        # rear axle -> centre of the deck, forward (m)
    blade_radius: float = 0.25       # radius of the circle swept by a blade tip (m)
    blade_lateral: tuple = (0.235, -0.235)   # y of each blade centre (left > 0)
    deck_margin: float = 0.03        # deck housing beyond the blade circles (m)
    # speeds (m/s) and time to change direction (s)
    v_mow: float = 1.2               # straight, forward
    v_turn: float = 0.7              # on an arc, forward
    v_reverse: float = 0.6           # any reverse motion
    t_shift: float = 2.0             # stop, change gear, start again
    max_steer_rate_deg: float = 40.0  # how fast the steering can turn (deg/s)

    def __post_init__(self):
        object.__setattr__(self, "blade_lateral", tuple(self.blade_lateral))
        if not 0 < self.max_steer_deg < 90:
            raise ValueError("max_steer_deg must be between 0 and 90 degrees")

    # ------------------------------------------------------------- derived
    @property
    def max_curvature(self) -> float:
        """kappa_max = tan(delta_max) / L (bicycle model)."""
        return math.tan(math.radians(self.max_steer_deg)) / self.wheelbase

    @property
    def min_radius(self) -> float:
        """Smallest turning radius of the middle of the rear axle, R = L / tan(delta_max)."""
        return 1.0 / self.max_curvature

    @property
    def cutting_width(self) -> float:
        """Width of the band cut when driving straight."""
        ys = np.asarray(self.blade_lateral)
        return float(ys.max() - ys.min() + 2 * self.blade_radius)

    @property
    def blade_centres(self) -> np.ndarray:
        """(k, 2) blade centres in the mower frame."""
        return np.array([(self.deck_offset, y) for y in self.blade_lateral])

    def steering_angle(self, curvature: float) -> float:
        """Angle of the virtual central front wheel for a path curvature (rad)."""
        return math.atan(self.wheelbase * curvature)

    def ackermann_angles(self, curvature: float) -> tuple[float, float]:
        """Angles of the left and right front wheels (rad). With Ackermann steering
        both wheels point at the same instantaneous centre of rotation."""
        if curvature == 0:
            return 0.0, 0.0
        r = 1.0 / curvature          # signed: > 0 for a left turn
        left = math.atan(self.wheelbase / (r - self.track / 2))
        right = math.atan(self.wheelbase / (r + self.track / 2))
        return left, right

    # --------------------------------------------------------------- shapes
    def body_outline(self) -> np.ndarray:
        """Chassis rectangle in the mower frame (counter-clockwise)."""
        x0, x1 = -self.rear_overhang, self.wheelbase + self.front_overhang
        h = self.body_width / 2
        return np.array([[x0, -h], [x1, -h], [x1, h], [x0, h]])

    def deck_outline(self) -> np.ndarray:
        """Deck housing rectangle in the mower frame."""
        ys = np.asarray(self.blade_lateral)
        r = self.blade_radius + self.deck_margin
        x0, x1 = self.deck_offset - r, self.deck_offset + r
        return np.array([[x0, ys.min() - r], [x1, ys.min() - r], [x1, ys.max() + r],
                         [x0, ys.max() + r]])

    def footprint_points(self, spacing: float = 0.05) -> np.ndarray:
        """Points on the outline of the body and of the deck housing (mower frame).
        The mower stays on the lawn if all of them do (for obstacles larger than
        ``spacing``)."""
        pts = []
        for poly in (self.body_outline(), self.deck_outline()):
            for a, b in zip(poly, np.roll(poly, -1, axis=0)):
                n = max(int(np.ceil(np.linalg.norm(b - a) / spacing)), 1)
                t = np.arange(n)[:, None] / n
                pts.append(a + t * (b - a))
        return np.vstack(pts)

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def from_file(path) -> "Mower":
        return Mower(**json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blade_lateral"] = list(self.blade_lateral)
        return d


def to_world(points: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    """Points given in the mower frame -> world frame, for a mower at pose (x, y, theta)."""
    c, s = math.cos(theta), math.sin(theta)
    p = np.asarray(points, dtype=float)
    return np.c_[x + c * p[..., 0] - s * p[..., 1], y + s * p[..., 0] + c * p[..., 1]]
