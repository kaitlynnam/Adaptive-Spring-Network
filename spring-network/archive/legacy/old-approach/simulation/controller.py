from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


TorqueFunction = Callable[[float], float]


@dataclass(frozen=True)
class FeedforwardTorqueController:
    """Controller that provides a required joint torque signal."""

    required_torque: TorqueFunction

    def compute_required_torque(self, t: float, theta: float, theta_dot: float) -> float:
        return float(self.required_torque(t))


@dataclass(frozen=True)
class TrackingPDController:
    """Simple desired-trajectory torque source for lightweight simulations."""

    desired_theta: TorqueFunction
    desired_theta_dot: TorqueFunction
    desired_theta_ddot: TorqueFunction
    inertia: float
    kp: float
    kd: float
    damping: float = 0.0

    def compute_required_torque(self, t: float, theta: float, theta_dot: float) -> float:
        theta_ref = self.desired_theta(t)
        theta_dot_ref = self.desired_theta_dot(t)
        theta_ddot_ref = self.desired_theta_ddot(t)
        return float(
            self.inertia * theta_ddot_ref
            + self.damping * theta_dot
            + self.kp * (theta_ref - theta)
            + self.kd * (theta_dot_ref - theta_dot)
        )
