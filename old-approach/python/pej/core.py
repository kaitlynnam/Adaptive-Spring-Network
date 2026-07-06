from __future__ import annotations

from dataclasses import dataclass
from math import inf, pi
from types import SimpleNamespace

import numpy as np


@dataclass
class PiecewiseProfile:
    active_theta: np.ndarray
    active_tau: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    ramp_margin_rad: float
    distillation_knots_theta: np.ndarray | None = None
    distillation_knots_tau: np.ndarray | None = None
    trim_fraction: float | None = None


@dataclass
class DistillationHistory:
    objective: np.ndarray
    gradient_norm: np.ndarray


@dataclass
class CamProfile:
    theta: np.ndarray
    tau_target: np.ndarray
    energy: np.ndarray
    radius: np.ndarray
    spring_deflection: np.ndarray
    spring_rate: float
    base_radius: float
    preload_deflection: float
    u0: float


def _as_1d(name: str, value: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return arr


def paper_tables() -> SimpleNamespace:
    """Numeric constants transcribed from the paper appendix."""

    tables = SimpleNamespace()
    tables.terrain_labels = ["Flat", "L1", "L2", "L3", "L4", "L5", "L6"]

    tables.cot_weights = SimpleNamespace(
        co_design=np.array([2.0, 1.0, 0.9, 0.5, 0.4, 0.2, 0.1]),
        reference=np.array([0.6, 0.6, 0.5, 0.4, 0.3, 0.15, 0.1]),
    )
    tables.power = SimpleNamespace(
        co_design=SimpleNamespace(
            before=np.array([21.7, 19.6, 24.3, 26.8, 31.0, 39.4, 49.5]),
            after=np.array([1.12, 3.70, 8.65, 16.6, 22.3, 32.3, 43.0]),
            offload_reported=np.array([94.8, 81.2, 64.4, 38.1, 28.0, 18.2, 13.1]),
        ),
        reference=SimpleNamespace(
            before=np.array([18.5, 21.3, 22.6, 25.4, 29.8, 38.2, 48.6]),
            after=np.array([15.4, 17.8, 18.5, 22.4, 25.2, 33.0, 43.0]),
            offload_reported=np.array([16.6, 16.3, 18.1, 11.7, 15.4, 13.6, 11.6]),
        ),
    )
    tables.reward_terms = [
        "track lin vel xy exp",
        "track ang vel z exp",
        "lin vel z l2",
        "ang vel xy l2",
        "base height above floor exp",
        "roll orientation exp",
        "pitch orientation exp",
        "dof torques l2",
        "dof acc l2",
        "action rate l2",
        "feet air time",
    ]
    tables.reward_weights = np.array(
        [2.5, 0.5, -2.0, -0.05, 0.5, 0.36, 0.09, -2e-4, -2.5e-7, -0.01, 0.01]
    )
    return tables


def residual_torque(tau_total: np.ndarray, tau_pej: np.ndarray) -> np.ndarray:
    """Active motor torque after passive PEJ assistance."""

    return np.asarray(tau_total, dtype=float) - np.asarray(tau_pej, dtype=float)


def motor_power(tau: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """Positive-only joint mechanical power."""

    return np.maximum(0.0, np.asarray(tau, dtype=float) * np.asarray(q_dot, dtype=float))


def residual_motor_power(
    tau_total: np.ndarray, tau_pej: np.ndarray, theta_dot: np.ndarray
) -> np.ndarray:
    """Positive residual motor power after PEJ subtraction."""

    return motor_power(residual_torque(tau_total, tau_pej), theta_dot)


def offload_percentage(power_without_pej: np.ndarray, power_with_pej: np.ndarray) -> np.ndarray:
    """PEJ power offload percentage."""

    return (
        (np.asarray(power_without_pej, dtype=float) - np.asarray(power_with_pej, dtype=float))
        / np.asarray(power_without_pej, dtype=float)
    ) * 100.0


def eval_piecewise_profile(
    theta: np.ndarray, knots_theta: np.ndarray, knots_tau: np.ndarray
) -> np.ndarray:
    """Evaluate a piecewise-linear PEJ torque-angle profile."""

    return np.interp(theta, knots_theta, knots_tau, left=0.0, right=0.0)


def make_piecewise_profile(
    theta_min: float,
    theta_max: float,
    knots_tau: np.ndarray,
    ramp_margin_rad: float = 5.0 * pi / 180.0,
) -> PiecewiseProfile:
    """Build a piecewise-linear PEJ profile with zero-torque ramp margins."""

    knots_tau = _as_1d("knots_tau", knots_tau)
    active_theta = np.linspace(theta_min, theta_max, knots_tau.size)
    theta = np.concatenate(([theta_min - ramp_margin_rad], active_theta, [theta_max + ramp_margin_rad]))
    tau = np.concatenate(([0.0], knots_tau, [0.0]))
    return PiecewiseProfile(
        active_theta=active_theta,
        active_tau=knots_tau,
        theta=theta,
        tau=tau,
        ramp_margin_rad=ramp_margin_rad,
    )


def projected_speed(v_actual: np.ndarray, v_cmd_unit: np.ndarray) -> np.ndarray:
    """Project actual velocity onto the desired direction."""

    v_actual = np.asarray(v_actual, dtype=float)
    v_cmd_unit = np.asarray(v_cmd_unit, dtype=float)
    if v_cmd_unit.ndim == 1:
        v_cmd_unit = np.broadcast_to(v_cmd_unit, v_actual.shape)
    return np.maximum(0.0, np.sum(v_actual * v_cmd_unit, axis=1))


def sliding_mean(x: np.ndarray, window_length: int = 10) -> np.ndarray:
    """Causal sliding mean used for smoothed projected speed."""

    if window_length <= 0:
        raise ValueError("window_length must be positive")
    x = _as_1d("x", x)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    y = np.empty_like(x)
    for i in range(x.size):
        start = max(0, i + 1 - window_length)
        y[i] = (cumsum[i + 1] - cumsum[start]) / (i + 1 - start)
    return y


def cost_of_transport(
    power_by_joint: np.ndarray,
    v_scalar: np.ndarray,
    mass: float = 15.0,
    gravity: float = 9.81,
    speed_floor: float = 0.1,
) -> np.ndarray:
    """Dimensionless mechanical Cost of Transport."""

    total_power = np.sum(np.asarray(power_by_joint, dtype=float), axis=1)
    v = np.maximum(np.asarray(v_scalar, dtype=float), speed_floor)
    return total_power / (mass * gravity * v)


def base_reward(weights: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Weighted reward sum."""

    return np.sum(np.asarray(weights, dtype=float) * np.asarray(rewards, dtype=float), axis=1)


def total_reward(r_base: np.ndarray, alpha: float, cot: np.ndarray) -> np.ndarray:
    """Stage-2 reward with CoT penalty."""

    return np.asarray(r_base, dtype=float) - alpha * np.asarray(cot, dtype=float)


def tracking_error(t: np.ndarray, v_cmd: np.ndarray, v_actual: np.ndarray) -> float:
    """Mean absolute velocity tracking error."""

    t = _as_1d("t", t)
    duration = t[-1] - t[0]
    if duration <= 0:
        raise ValueError("t must span a positive duration")
    return float(np.trapezoid(np.abs(np.asarray(v_cmd, dtype=float) - np.asarray(v_actual, dtype=float)), t) / duration)


def cam_radius_from_torque(
    theta: np.ndarray,
    tau_target: np.ndarray,
    spring_rate: float,
    base_radius: float = 0.050,
    preload_deflection: float = 0.0,
    u0: float | None = None,
) -> CamProfile:
    """Map target torque to cam radius profile."""

    theta = _as_1d("theta", theta)
    tau_target = _as_1d("tau_target", tau_target)
    if theta.shape != tau_target.shape:
        raise ValueError("theta and tau_target must have the same shape")
    if spring_rate <= 0 or base_radius <= 0:
        raise ValueError("spring_rate and base_radius must be positive")
    if preload_deflection < 0:
        raise ValueError("preload_deflection must be nonnegative")

    dx = np.diff(theta)
    avg_y = 0.5 * (tau_target[:-1] + tau_target[1:])
    energy_integral = np.concatenate(([0.0], np.cumsum(dx * avg_y)))
    u0_value = float(np.max(energy_integral) if u0 is None else u0)
    energy = np.maximum(u0_value - energy_integral, 0.0)
    spring_deflection = np.sqrt(2.0 * energy / spring_rate)
    radius = base_radius + spring_deflection - preload_deflection
    return CamProfile(
        theta=theta,
        tau_target=tau_target,
        energy=energy,
        radius=radius,
        spring_deflection=spring_deflection,
        spring_rate=spring_rate,
        base_radius=base_radius,
        preload_deflection=preload_deflection,
        u0=u0_value,
    )


def distill_profile(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    tau_total: np.ndarray,
    *,
    num_knots: int = 20,
    trim_fraction: float = 0.05,
    learning_rate: float = 0.15,
    momentum: float = 0.8,
    max_iterations: int = 500,
    gradient_tolerance: float = 1e-8,
    ramp_margin_rad: float = 5.0 * pi / 180.0,
    initial_tau: np.ndarray | None = None,
    torque_limit: float = inf,
) -> tuple[PiecewiseProfile, DistillationHistory]:
    """Distill a PEJ torque-angle profile from joint data."""

    theta = _as_1d("theta", theta)
    theta_dot = _as_1d("theta_dot", theta_dot)
    tau_total = _as_1d("tau_total", tau_total)
    if theta.shape != theta_dot.shape or theta.shape != tau_total.shape:
        raise ValueError("theta, theta_dot, and tau_total must have the same shape")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if num_knots <= 0 or max_iterations <= 0:
        raise ValueError("num_knots and max_iterations must be positive")

    theta_use, theta_dot_use, tau_total_use, theta_min, theta_max = _trim_by_angle(
        theta, theta_dot, tau_total, trim_fraction
    )
    knots_theta = np.linspace(theta_min, theta_max, num_knots)
    if initial_tau is None:
        knots_tau = np.zeros(num_knots)
    else:
        knots_tau = _as_1d("initial_tau", initial_tau).copy()
        if knots_tau.size != num_knots:
            raise ValueError("initial_tau must have num_knots values")

    velocity = np.zeros(num_knots)
    objective_values: list[float] = []
    gradient_norms: list[float] = []

    for _ in range(max_iterations):
        objective, grad = _objective_and_gradient(theta_use, theta_dot_use, tau_total_use, knots_theta, knots_tau)
        grad_norm = float(np.linalg.norm(grad, ord=2))
        objective_values.append(objective)
        gradient_norms.append(grad_norm)

        if grad_norm <= gradient_tolerance:
            break

        velocity = momentum * velocity - learning_rate * grad
        knots_tau = np.clip(knots_tau + velocity, -torque_limit, torque_limit)

    profile = make_piecewise_profile(theta_min, theta_max, knots_tau, ramp_margin_rad)
    profile.distillation_knots_theta = knots_theta
    profile.distillation_knots_tau = knots_tau
    profile.trim_fraction = trim_fraction
    history = DistillationHistory(np.array(objective_values), np.array(gradient_norms))
    return profile, history


def _trim_by_angle(
    theta: np.ndarray, theta_dot: np.ndarray, tau_total: np.ndarray, trim_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    theta_min = float(np.quantile(theta, trim_fraction, method="linear"))
    theta_max = float(np.quantile(theta, 1.0 - trim_fraction, method="linear"))
    mask = (theta >= theta_min) & (theta <= theta_max)
    return theta[mask], theta_dot[mask], tau_total[mask], theta_min, theta_max


def _objective_and_gradient(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    tau_total: np.ndarray,
    knots_theta: np.ndarray,
    knots_tau: np.ndarray,
) -> tuple[float, np.ndarray]:
    tau_pej = np.interp(theta, knots_theta, knots_tau)
    raw_power = (tau_total - tau_pej) * theta_dot
    active = raw_power > 0.0
    objective = float(np.sum(raw_power[active]))

    grad = np.zeros(knots_theta.size)
    if not np.any(active):
        return objective, grad

    left_idx, right_idx, left_weight, right_weight = _interpolation_weights(theta[active], knots_theta)
    contrib = -theta_dot[active]
    np.add.at(grad, left_idx, contrib * left_weight)
    np.add.at(grad, right_idx, contrib * right_weight)
    return objective, grad


def _interpolation_weights(theta: np.ndarray, knots_theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = knots_theta.size
    dtheta = knots_theta[1] - knots_theta[0]
    right_idx = np.ceil((theta - knots_theta[0]) / dtheta).astype(int)
    right_idx = np.clip(right_idx, 1, n - 1)
    left_idx = right_idx - 1
    right_weight = np.clip((theta - knots_theta[left_idx]) / dtheta, 0.0, 1.0)
    left_weight = 1.0 - right_weight
    return left_idx, right_idx, left_weight, right_weight
