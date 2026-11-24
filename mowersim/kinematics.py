"""Kinematic bicycle model and trajectories made of arcs and straight lines.

The state is the pose (x, y, theta) of the middle of the rear axle. With no
wheel slip, this point moves along the heading, and its path has curvature

    kappa = tan(delta) / L                 (delta: steering angle, L: wheelbase)

so a constant steering angle gives an arc of a circle (a straight line when
delta = 0). Driving a distance s along the path (s < 0 in reverse):

    theta(s) = theta0 + kappa s
    x(s) = x0 + (sin theta(s) - sin theta0) / kappa
    y(s) = y0 - (cos theta(s) - cos theta0) / kappa        (kappa != 0)

A trajectory is a start pose and a list of segments, each with a constant
curvature, a length, a direction (forward / reverse) and the blades on or off.
Every path the planners produce has this form, which keeps the motion exact
(no numerical integration error).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Pose


@dataclass(frozen=True)
class Segment:
    curvature: float                 # 1/m: tan(delta)/L, > 0 with the front wheels turned
                                     # left, 0 = straight (same sign forward and in reverse)
    length: float                    # distance travelled by the rear-axle point (m), >= 0
    reverse: bool = False
    mow: bool = True                 # blades engaged

    @property
    def signed_length(self) -> float:
        return -self.length if self.reverse else self.length


def advance(pose: Pose, curvature: float, s: float) -> Pose:
    """Exact pose after driving a signed distance s on an arc of the given curvature."""
    x, y, th = pose
    if abs(curvature) < 1e-12:
        return Pose(x + s * math.cos(th), y + s * math.sin(th), th)
    th1 = th + curvature * s
    return Pose(x + (math.sin(th1) - math.sin(th)) / curvature,
                y - (math.cos(th1) - math.cos(th)) / curvature, th1)


def advance_many(x, y, th, curvature, s):
    """Vectorised version of ``advance`` (numpy arrays, same shapes)."""
    k = np.asarray(curvature, dtype=float)
    straight = np.abs(k) < 1e-12
    ks = np.where(straight, 1.0, k)
    th1 = th + k * s
    x1 = np.where(straight, x + s * np.cos(th), x + (np.sin(th1) - np.sin(th)) / ks)
    y1 = np.where(straight, y + s * np.sin(th), y - (np.cos(th1) - np.cos(th)) / ks)
    return x1, y1, th1


@dataclass
class Samples:
    """A trajectory sampled along its length (one row per sample)."""
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    curvature: np.ndarray
    reverse: np.ndarray
    mow: np.ndarray
    segment: np.ndarray              # index of the segment each sample belongs to
    s: np.ndarray                    # distance travelled since the start (m, always >= 0)


@dataclass
class Trajectory:
    start: Pose
    segments: list[Segment] = field(default_factory=list)

    # ---------------------------------------------------------- building
    def then(self, curvature: float, length: float, reverse: bool = False,
             mow: bool = True) -> "Trajectory":
        if length < 0:
            raise ValueError("segment length must be >= 0; use reverse=True to back up")
        if length > 0:
            self.segments.append(Segment(curvature, length, reverse, mow))
        return self

    def extend(self, segments) -> "Trajectory":
        for seg in segments:
            self.then(seg.curvature, seg.length, seg.reverse, seg.mow)
        return self

    # ------------------------------------------------------------ queries
    def poses(self) -> list[Pose]:
        """Pose at the start and at the end of every segment."""
        out = [self.start]
        for seg in self.segments:
            out.append(advance(out[-1], seg.curvature, seg.signed_length))
        return out

    @property
    def end(self) -> Pose:
        return self.poses()[-1]

    @property
    def length(self) -> float:
        return sum(seg.length for seg in self.segments)

    def direction_changes(self) -> int:
        """Number of cusps: places where the mower stops and changes gear."""
        moving = [seg.reverse for seg in self.segments if seg.length > 0]
        return sum(a != b for a, b in zip(moving, moving[1:]))

    def max_curvature(self) -> float:
        return max((abs(seg.curvature) for seg in self.segments), default=0.0)

    def sample(self, step: float) -> Samples:
        """Sample the trajectory every ``step`` metres (and at every segment end)."""
        rows = {k: [] for k in ("x", "y", "theta", "curvature", "reverse", "mow", "segment",
                                "s")}
        pose, s0 = self.start, 0.0
        for i, seg in enumerate(self.segments):
            n = max(int(math.ceil(seg.length / step)), 1)
            d = np.linspace(0.0, seg.length, n + 1)[:-1] if i < len(self.segments) - 1 \
                else np.linspace(0.0, seg.length, n + 1)
            sign = -1.0 if seg.reverse else 1.0
            x, y, th = advance_many(pose.x, pose.y, pose.theta, seg.curvature, sign * d)
            rows["x"].append(x)
            rows["y"].append(y)
            rows["theta"].append(th)
            m = len(d)
            rows["curvature"].append(np.full(m, seg.curvature))
            rows["reverse"].append(np.full(m, seg.reverse))
            rows["mow"].append(np.full(m, seg.mow))
            rows["segment"].append(np.full(m, i))
            rows["s"].append(s0 + d)
            pose = advance(pose, seg.curvature, seg.signed_length)
            s0 += seg.length
        if not self.segments:
            return Samples(*(np.array([v]) for v in (self.start.x, self.start.y,
                                                     self.start.theta, 0.0, False, False, -1,
                                                     0.0)))
        return Samples(**{k: np.concatenate(v) for k, v in rows.items()})

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {"start": [self.start.x, self.start.y, math.degrees(self.start.theta)],
                "segments": [asdict(seg) for seg in self.segments]}

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        x, y, deg = d["start"]
        return Trajectory(Pose(x, y, math.radians(deg)), [Segment(**s) for s in d["segments"]])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @staticmethod
    def load(path) -> "Trajectory":
        return Trajectory.from_dict(json.loads(Path(path).read_text()))
