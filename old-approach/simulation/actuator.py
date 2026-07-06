from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActuatorState:
    q: np.ndarray
    phi: np.ndarray
    k_eff: np.ndarray


@dataclass(frozen=True)
class CamActuatorConfig:
    tau_response: float
    max_speed: float

    def __post_init__(self) -> None:
        if self.tau_response <= 0.0:
            raise ValueError("tau_response must be positive")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")


@dataclass(frozen=True)
class CamActuatorStep:
    phi: float
    phi_dot: float


def actuator_command(
    q: np.ndarray | float,
    *,
    k_soft: float,
    k_stiff: float,
    phi_min: float,
    phi_max: float,
) -> ActuatorState:
    """Map tuning command q to actuator angle and effective stiffness."""

    if k_stiff < k_soft:
        raise ValueError("k_stiff must be greater than or equal to k_soft")
    q_clamped = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    phi = phi_min + q_clamped * (phi_max - phi_min)
    k_eff = k_soft + q_clamped * (k_stiff - k_soft)
    return ActuatorState(q=q_clamped, phi=phi, k_eff=k_eff)


def actuator_state_derivative(phi: float, phi_command: float, time_constant: float) -> float:
    """First-order tuning actuator response: dphi/dt = (phi_cmd - phi) / tau."""

    if time_constant <= 0.0:
        raise ValueError("time_constant must be positive")
    return float((phi_command - phi) / time_constant)


def stiffness_from_phi(
    phi: np.ndarray | float,
    *,
    k_soft: float,
    k_stiff: float,
    phi_min: float,
    phi_max: float,
) -> np.ndarray:
    """Recover effective stiffness from physical actuator angle."""

    if phi_max <= phi_min:
        raise ValueError("phi_max must be greater than phi_min")
    q = (np.asarray(phi, dtype=float) - phi_min) / (phi_max - phi_min)
    q = np.clip(q, 0.0, 1.0)
    return k_soft + q * (k_stiff - k_soft)


def update_cam_actuator(
    phi: float,
    phi_desired: float,
    dt: float,
    config: CamActuatorConfig,
) -> CamActuatorStep:
    """Speed-limited first-order cam actuator update."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    phi_dot_command = (phi_desired - phi) / config.tau_response
    phi_dot = float(np.clip(phi_dot_command, -config.max_speed, config.max_speed))
    return CamActuatorStep(phi=float(phi + phi_dot * dt), phi_dot=phi_dot)
