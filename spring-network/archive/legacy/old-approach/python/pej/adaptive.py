from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .core import PiecewiseProfile, eval_piecewise_profile


ProfileLike = Callable[[np.ndarray], np.ndarray] | PiecewiseProfile


@dataclass
class ActuatorTunedResponse:
    q: np.ndarray
    phi: np.ndarray
    k_eff: np.ndarray
    tau_spring: np.ndarray


def blend_profiles(
    theta: np.ndarray | float,
    flat_profile: ProfileLike,
    rough_profile: ProfileLike,
    q: np.ndarray | float,
) -> np.ndarray:
    """
    Return adaptive spring torque by blending two spring profiles.

    q = 0 gives flat_profile, q = 1 gives rough_profile, and intermediate
    values linearly interpolate between the two.
    """

    theta_values = np.asarray(theta, dtype=float)
    q_values = np.asarray(q, dtype=float)
    flat_tau = _evaluate_profile(flat_profile, theta_values)
    rough_tau = _evaluate_profile(rough_profile, theta_values)
    return (1.0 - q_values) * flat_tau + q_values * rough_tau


def roughness_score(theta_dot_window: np.ndarray) -> float:
    """
    Estimate terrain roughness from recent joint velocity variation.

    Higher variance means rougher motion.
    """

    theta_dot_values = np.asarray(theta_dot_window, dtype=float)
    if theta_dot_values.size == 0:
        raise ValueError("theta_dot_window must contain at least one sample")
    return float(np.var(theta_dot_values))


def roughness_to_q(
    score: np.ndarray | float,
    min_score: float,
    max_score: float,
) -> np.ndarray:
    """Convert roughness score into a tuning value between 0 and 1."""

    if max_score <= min_score:
        raise ValueError("max_score must be greater than min_score")
    q = (np.asarray(score, dtype=float) - min_score) / (max_score - min_score)
    return np.clip(q, 0.0, 1.0)


def actuator_tuned_stiffness(
    theta: np.ndarray | float,
    q: np.ndarray | float,
    *,
    k_soft: float,
    k_stiff: float,
    phi_min: float,
    phi_max: float,
    actuator_time_constant: float | None = None,
    dt: float | None = None,
) -> ActuatorTunedResponse:
    """
    Compute torque from a software-sensed, actuator-tuned PEJ stiffness model.

    q commands a virtual tuning actuator. The actuator changes effective PEJ
    stiffness but does not directly power the joint.
    """

    if k_stiff < k_soft:
        raise ValueError("k_stiff must be greater than or equal to k_soft")
    if actuator_time_constant is not None and actuator_time_constant <= 0.0:
        raise ValueError("actuator_time_constant must be positive")
    if actuator_time_constant is not None and dt is not None and dt <= 0.0:
        raise ValueError("dt must be positive")

    theta_values = np.asarray(theta, dtype=float)
    q_values = clamp_q(q)
    if actuator_time_constant is not None and dt is not None:
        q_values = _first_order_lag(q_values, dt, actuator_time_constant)

    phi = phi_min + q_values * (phi_max - phi_min)
    k_eff = k_soft + q_values * (k_stiff - k_soft)
    tau_spring = k_eff * theta_values
    return ActuatorTunedResponse(q=q_values, phi=phi, k_eff=k_eff, tau_spring=tau_spring)


def clamp_q(q: np.ndarray | float) -> np.ndarray:
    """Clamp a tuning command into the physically valid range [0, 1]."""

    return np.clip(np.asarray(q, dtype=float), 0.0, 1.0)


def _evaluate_profile(profile: ProfileLike, theta: np.ndarray) -> np.ndarray:
    if isinstance(profile, PiecewiseProfile):
        return eval_piecewise_profile(theta, profile.theta, profile.tau)
    return np.asarray(profile(theta), dtype=float)


def _first_order_lag(q: np.ndarray, dt: float, time_constant: float) -> np.ndarray:
    if q.ndim == 0:
        return q

    alpha = dt / (time_constant + dt)
    filtered = np.empty_like(q, dtype=float)
    filtered[0] = q[0]
    for i in range(1, q.size):
        filtered[i] = filtered[i - 1] + alpha * (q[i] - filtered[i - 1])
    return filtered
