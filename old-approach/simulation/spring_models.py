from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from python.pej.adaptive import actuator_tuned_stiffness, blend_profiles
from python.pej.core import PiecewiseProfile

from .actuator import actuator_command, stiffness_from_phi


ProfileFunction = Callable[[np.ndarray], np.ndarray] | PiecewiseProfile


@dataclass(frozen=True)
class FixedSpringModel:
    """Linear fixed-stiffness spring: tau = k * (theta - rest_angle)."""

    stiffness: float
    rest_angle: float = 0.0

    def compute_torque(self, theta: np.ndarray | float, q: np.ndarray | float = 0.0) -> np.ndarray:
        return self.stiffness * (np.asarray(theta, dtype=float) - self.rest_angle)

    def effective_stiffness(self, q: np.ndarray | float = 0.0) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.stiffness, dtype=float), np.asarray(q, dtype=float).shape)


@dataclass(frozen=True)
class AdaptiveBlendModel:
    """Blend two torque-angle profiles using tuning state q."""

    flat_profile: ProfileFunction
    rough_profile: ProfileFunction

    def compute_torque(self, theta: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
        return blend_profiles(theta, self.flat_profile, self.rough_profile, np.clip(q, 0.0, 1.0))

    def effective_stiffness(self, q: np.ndarray | float = 0.0) -> np.ndarray:
        return np.full_like(np.asarray(q, dtype=float), np.nan, dtype=float)


@dataclass(frozen=True)
class ActuatorTunedModel:
    """Software-sensed, actuator-tuned spring stiffness model."""

    k_soft: float
    k_stiff: float
    phi_min: float
    phi_max: float
    rest_angle: float = 0.0
    actuator_time_constant: float | None = None

    def compute_torque(
        self,
        theta: np.ndarray | float,
        q: np.ndarray | float,
        phi: np.ndarray | float | None = None,
    ) -> np.ndarray:
        theta_offset = np.asarray(theta, dtype=float) - self.rest_angle
        if phi is None:
            return actuator_tuned_stiffness(
                theta_offset,
                q,
                k_soft=self.k_soft,
                k_stiff=self.k_stiff,
                phi_min=self.phi_min,
                phi_max=self.phi_max,
            ).tau_spring
        k_eff = stiffness_from_phi(
            phi,
            k_soft=self.k_soft,
            k_stiff=self.k_stiff,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
        )
        return k_eff * theta_offset

    def command(self, q: np.ndarray | float):
        return actuator_command(
            q,
            k_soft=self.k_soft,
            k_stiff=self.k_stiff,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
        )

    def effective_stiffness(
        self,
        q: np.ndarray | float,
        phi: np.ndarray | float | None = None,
    ) -> np.ndarray:
        if phi is not None:
            return stiffness_from_phi(
                phi,
                k_soft=self.k_soft,
                k_stiff=self.k_stiff,
                phi_min=self.phi_min,
                phi_max=self.phi_max,
            )
        return self.command(q).k_eff
