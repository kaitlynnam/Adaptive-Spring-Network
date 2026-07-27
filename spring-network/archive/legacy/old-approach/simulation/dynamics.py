from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from python.pej.core import motor_power

from .actuator import actuator_state_derivative
from .spring_models import ActuatorTunedModel


QFunction = Callable[[float], float]


@dataclass(frozen=True)
class JointSimulationConfig:
    inertia: float = 0.05
    damping: float = 0.0
    t_span: tuple[float, float] = (0.0, 5.0)
    dt: float = 0.01
    initial_theta: float = 0.0
    initial_theta_dot: float = 0.0
    initial_phi: float | None = None
    rtol: float = 1e-7
    atol: float = 1e-9


@dataclass(frozen=True)
class JointSimulationResult:
    time: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray
    q: np.ndarray
    phi: np.ndarray
    k_eff: np.ndarray
    tau_required: np.ndarray
    tau_spring: np.ndarray
    tau_motor: np.ndarray
    motor_power: np.ndarray


def simulate_joint(
    *,
    spring_model,
    controller,
    q_input: QFunction,
    config: JointSimulationConfig,
) -> JointSimulationResult:
    """Integrate adaptive PEJ joint dynamics with scipy.integrate.solve_ivp."""

    if config.inertia <= 0.0:
        raise ValueError("inertia must be positive")
    if config.dt <= 0.0:
        raise ValueError("dt must be positive")
    t0, tf = config.t_span
    if tf <= t0:
        raise ValueError("t_span must have positive duration")

    uses_phi_state = (
        isinstance(spring_model, ActuatorTunedModel)
        and spring_model.actuator_time_constant is not None
    )
    initial_state = [config.initial_theta, config.initial_theta_dot]
    if uses_phi_state:
        if config.initial_phi is None:
            initial_phi = float(spring_model.command(q_input(t0)).phi)
        else:
            initial_phi = config.initial_phi
        initial_state.append(initial_phi)

    def rhs(t: float, state: np.ndarray) -> list[float]:
        theta = float(state[0])
        theta_dot = float(state[1])
        q = float(np.clip(q_input(t), 0.0, 1.0))
        phi = float(state[2]) if uses_phi_state else None
        tau_spring = float(_spring_torque(spring_model, theta, q, phi))
        tau_required = float(controller.compute_required_torque(t, theta, theta_dot))
        tau_motor = tau_required - tau_spring
        theta_ddot = (tau_motor + tau_spring - config.damping * theta_dot) / config.inertia
        derivatives = [theta_dot, theta_ddot]
        if uses_phi_state:
            phi_cmd = float(spring_model.command(q).phi)
            derivatives.append(
                actuator_state_derivative(
                    phi,
                    phi_cmd,
                    spring_model.actuator_time_constant,
                )
            )
        return derivatives

    t_eval = np.arange(t0, tf + 0.5 * config.dt, config.dt)
    solution = solve_ivp(
        rhs,
        (t0, tf),
        initial_state,
        t_eval=t_eval,
        rtol=config.rtol,
        atol=config.atol,
    )
    if not solution.success:
        raise RuntimeError(f"joint simulation failed: {solution.message}")

    time = solution.t
    theta = solution.y[0]
    theta_dot = solution.y[1]
    q = np.clip(np.array([q_input(t) for t in time], dtype=float), 0.0, 1.0)
    phi = _phi_trace(spring_model, q, solution.y[2] if uses_phi_state else None)
    k_eff = _stiffness_trace(spring_model, q, phi if uses_phi_state else None)
    tau_spring = _spring_torque(spring_model, theta, q, phi if uses_phi_state else None)
    tau_required = np.array(
        [controller.compute_required_torque(t, th, thd) for t, th, thd in zip(time, theta, theta_dot)],
        dtype=float,
    )
    tau_motor = tau_required - tau_spring
    power = motor_power(tau_motor, theta_dot)

    return JointSimulationResult(
        time=time,
        theta=theta,
        theta_dot=theta_dot,
        q=q,
        phi=phi,
        k_eff=k_eff,
        tau_required=tau_required,
        tau_spring=tau_spring,
        tau_motor=tau_motor,
        motor_power=power,
    )


def _spring_torque(spring_model, theta, q, phi=None):
    if isinstance(spring_model, ActuatorTunedModel):
        return spring_model.compute_torque(theta, q, phi=phi)
    return spring_model.compute_torque(theta, q)


def _phi_trace(spring_model, q: np.ndarray, phi_state: np.ndarray | None) -> np.ndarray:
    if isinstance(spring_model, ActuatorTunedModel):
        if phi_state is not None:
            return np.asarray(phi_state, dtype=float)
        return np.asarray(spring_model.command(q).phi, dtype=float)
    return np.full_like(q, np.nan, dtype=float)


def _stiffness_trace(spring_model, q: np.ndarray, phi: np.ndarray | None) -> np.ndarray:
    if hasattr(spring_model, "effective_stiffness"):
        if isinstance(spring_model, ActuatorTunedModel):
            return np.asarray(spring_model.effective_stiffness(q, phi=phi), dtype=float)
        return np.asarray(spring_model.effective_stiffness(q), dtype=float)
    return np.full_like(q, np.nan, dtype=float)
