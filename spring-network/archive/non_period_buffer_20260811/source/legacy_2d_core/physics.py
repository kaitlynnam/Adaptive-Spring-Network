from dataclasses import dataclass

import numpy as np


@dataclass
class Spring:
    node_a: str
    node_b: str
    stiffness_k: float
    rest_length: float | None = None


def spring_force(position_a, position_b, stiffness_k, rest_length):
    """Return the force applied to node A by a Hookean spring.

    The spring direction points from node A to node B. Positive stretch means
    the spring is longer than its rest length, so node A is pulled toward node B.
    """
    delta = position_b - position_a
    current_length = np.linalg.norm(delta)

    if current_length < 1e-12:
        return np.zeros(2), current_length, 0.0

    direction = delta / current_length
    stretch = current_length - rest_length
    force_on_a = stiffness_k * stretch * direction
    return force_on_a, current_length, stretch


def torque_about_origin(position, force):
    """Compute 2D torque tau = r_x * F_y - r_y * F_x."""
    return position[0] * force[1] - position[1] * force[0]
