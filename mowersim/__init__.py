"""mowersim: coverage path planning for a car-like robot mower.

Polygon lawns in metres, a mower that steers like a car and can reverse
(Reeds-Shepp paths), and coverage measured from the area swept by the blades.
"""
from .coverage import Result, effective_width, evaluate, length_lower_bound
from .geometry import Lawn, Pose, example_lawns
from .kinematics import Segment, Trajectory, advance
from .mower import Mower
from .paths import candidates, fastest, shortest

__all__ = ["Lawn", "Pose", "example_lawns", "Mower", "Segment", "Trajectory", "advance",
           "candidates", "shortest", "fastest", "evaluate", "Result", "effective_width",
           "length_lower_bound"]
