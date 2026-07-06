from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear, minimize

from python.pej.core import residual_motor_power


@dataclass(frozen=True)
class CamSpring:
    k: float
    preload: float
    lever_arm: float
    phi_start: float
    phi_end: float
    max_compression: float

    def __post_init__(self) -> None:
        if self.k < 0.0:
            raise ValueError("spring constant k must be nonnegative")
        if self.preload < 0.0:
            raise ValueError("preload must be nonnegative")
        if self.lever_arm <= 0.0:
            raise ValueError("lever_arm must be positive")
        if self.phi_end <= self.phi_start:
            raise ValueError("phi_end must be greater than phi_start")
        if self.max_compression < 0.0:
            raise ValueError("max_compression must be nonnegative")


@dataclass(frozen=True)
class CamSpringNetwork:
    springs: list[CamSpring]
    theta_rest: float = 0.0
    assist_direction: float = 1.0

    def __post_init__(self) -> None:
        if not self.springs:
            raise ValueError("CamSpringNetwork requires at least one spring")
        if self.assist_direction not in (-1.0, 1.0):
            raise ValueError("assist_direction must be 1.0 or -1.0")

    def torque(self, theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
        """Restoring passive spring torque from the cam-controlled network."""

        forces = self.spring_forces(theta, phi)
        lever_arms = np.array([spring.lever_arm for spring in self.springs], dtype=float)
        return -self.assist_direction * np.sum(forces * _expand_spring_axis(lever_arms, forces.ndim), axis=0)

    def spring_forces(self, theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
        compressions = self.spring_compressions(theta, phi)
        spring_rates = np.array([spring.k for spring in self.springs], dtype=float)
        return compressions * _expand_spring_axis(spring_rates, compressions.ndim)

    def spring_compressions(self, theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
        theta_values = np.asarray(theta, dtype=float)
        phi_values = np.asarray(phi, dtype=float)
        compressions = []
        for spring in self.springs:
            engagement = smooth_engagement(phi_values, spring.phi_start, spring.phi_end)
            joint_compression = spring.lever_arm * self.assist_direction * (theta_values - self.theta_rest)
            compression = spring.preload + engagement * spring.max_compression + joint_compression
            compressions.append(np.maximum(0.0, compression))
        return np.stack(compressions, axis=0)

    def spring_potential_energy(self, theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
        compressions = self.spring_compressions(theta, phi)
        spring_rates = np.array([spring.k for spring in self.springs], dtype=float)
        return 0.5 * np.sum(_expand_spring_axis(spring_rates, compressions.ndim) * compressions**2, axis=0)

    def cam_torque(self, theta: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
        """Approximate actuator torque needed to move the cam against spring forces."""

        forces = self.spring_forces(theta, phi)
        phi_values = np.asarray(phi, dtype=float)
        derivatives = []
        for spring in self.springs:
            inside_engagement = (phi_values >= spring.phi_start) & (phi_values <= spring.phi_end)
            slope = spring.max_compression / (spring.phi_end - spring.phi_start)
            derivatives.append(np.where(inside_engagement, slope, 0.0))
        dxi_dphi = np.stack(derivatives, axis=0)
        return np.sum(forces * dxi_dphi, axis=0)

    def effective_stiffness_numeric(self, theta: float, phi: float, eps: float = 1e-5) -> float:
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        tau_plus = self.torque(theta + eps, phi)
        tau_minus = self.torque(theta - eps, phi)
        return float((tau_plus - tau_minus) / (2.0 * eps))

    def cam_torque_energy_error(self, theta: float, phi: float, eps: float = 1e-5) -> float:
        """Compare cam torque estimate against dU/dphi by finite difference."""

        if eps <= 0.0:
            raise ValueError("eps must be positive")
        u_plus = self.spring_potential_energy(theta, phi + eps)
        u_minus = self.spring_potential_energy(theta, phi - eps)
        dU_dphi = float((u_plus - u_minus) / (2.0 * eps))
        return float(self.cam_torque(theta, phi) - dU_dphi)

    def check_constraints(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        *,
        phi_dot: np.ndarray | None = None,
        limits: "CamSpringNetworkLimits",
    ) -> "CamSpringNetworkConstraintReport":
        compressions = self.spring_compressions(theta, phi)
        forces = self.spring_forces(theta, phi)
        torque = self.torque(theta, phi)
        cam_torque = self.cam_torque(theta, phi)
        max_phi_speed = 0.0 if phi_dot is None else float(np.max(np.abs(phi_dot)))
        report = CamSpringNetworkConstraintReport(
            max_compression=float(np.max(compressions)),
            max_force=float(np.max(forces)),
            max_abs_spring_torque=float(np.max(np.abs(torque))),
            max_abs_cam_torque=float(np.max(np.abs(cam_torque))),
            max_abs_phi_speed=max_phi_speed,
            limits=limits,
        )
        return report


@dataclass(frozen=True)
class CamSpringNetworkModel:
    """Spring-model adapter that maps q to cam angle and then cam torque."""

    network: CamSpringNetwork
    phi_min: float = 0.0
    phi_max: float = 1.2

    def compute_torque(self, theta: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
        phi = self.phi_from_q(q)
        return self.network.torque(theta, phi)

    def phi_from_q(self, q: np.ndarray | float) -> np.ndarray:
        q_clamped = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        return self.phi_min + q_clamped * (self.phi_max - self.phi_min)

    def effective_stiffness(self, q: np.ndarray | float = 0.0) -> np.ndarray:
        phi = self.phi_from_q(q)
        return np.array([self.network.effective_stiffness_numeric(self.network.theta_rest, float(value)) for value in np.ravel(phi)]).reshape(
            np.asarray(phi).shape
        )


@dataclass(frozen=True)
class CamSpringNetworkLimits:
    max_compression: float
    max_force: float
    max_abs_spring_torque: float
    max_abs_cam_torque: float
    max_abs_phi_speed: float


@dataclass(frozen=True)
class CamSpringNetworkConstraintReport:
    max_compression: float
    max_force: float
    max_abs_spring_torque: float
    max_abs_cam_torque: float
    max_abs_phi_speed: float
    limits: CamSpringNetworkLimits

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def violations(self) -> list[str]:
        violations = []
        if _exceeds_limit(self.max_compression, self.limits.max_compression):
            violations.append("max_compression")
        if _exceeds_limit(self.max_force, self.limits.max_force):
            violations.append("max_force")
        if _exceeds_limit(self.max_abs_spring_torque, self.limits.max_abs_spring_torque):
            violations.append("max_abs_spring_torque")
        if _exceeds_limit(self.max_abs_cam_torque, self.limits.max_abs_cam_torque):
            violations.append("max_abs_cam_torque")
        if _exceeds_limit(self.max_abs_phi_speed, self.limits.max_abs_phi_speed):
            violations.append("max_abs_phi_speed")
        return violations


@dataclass(frozen=True)
class CamSpringFitResult:
    network: CamSpringNetwork
    fitted_spring_rates: np.ndarray
    rms_error: float
    max_abs_error: float
    predicted_torque: np.ndarray


@dataclass(frozen=True)
class CamSpringEnergyOptimizationResult:
    network: CamSpringNetwork
    optimized_spring_rates: np.ndarray
    objective_energy: float
    motor_energy: float
    cam_actuator_energy: float
    net_energy_saved: float
    net_offload_percentage: float
    success: bool
    message: str


def fit_spring_rates_to_targets(
    template_network: CamSpringNetwork,
    theta: np.ndarray,
    phi: np.ndarray,
    target_torque: np.ndarray,
    *,
    min_k: float = 0.0,
    max_k: float = np.inf,
) -> CamSpringFitResult:
    """Fit spring constants so the cam network approximates target torque samples."""

    theta_values = np.asarray(theta, dtype=float)
    phi_values = np.asarray(phi, dtype=float)
    target_values = np.asarray(target_torque, dtype=float)
    if theta_values.shape != phi_values.shape or theta_values.shape != target_values.shape:
        raise ValueError("theta, phi, and target_torque must have the same shape")

    basis = _torque_basis_for_spring_rates(template_network, theta_values, phi_values)
    solution = lsq_linear(basis, target_values, bounds=(min_k, max_k))
    fitted_springs = [
        CamSpring(
            k=float(k),
            preload=spring.preload,
            lever_arm=spring.lever_arm,
            phi_start=spring.phi_start,
            phi_end=spring.phi_end,
            max_compression=spring.max_compression,
        )
        for k, spring in zip(solution.x, template_network.springs)
    ]
    fitted_network = CamSpringNetwork(
        fitted_springs,
        theta_rest=template_network.theta_rest,
        assist_direction=template_network.assist_direction,
    )
    predicted = fitted_network.torque(theta_values, phi_values)
    error = predicted - target_values
    return CamSpringFitResult(
        network=fitted_network,
        fitted_spring_rates=np.asarray(solution.x, dtype=float),
        rms_error=float(np.sqrt(np.mean(error**2))),
        max_abs_error=float(np.max(np.abs(error))),
        predicted_torque=predicted,
    )


def optimize_spring_rates_for_energy(
    template_network: CamSpringNetwork,
    time: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    phi: np.ndarray,
    phi_dot: np.ndarray,
    tau_required: np.ndarray,
    motor_power_without_spring: np.ndarray,
    *,
    initial_k: np.ndarray | None = None,
    min_k: float = 0.0,
    max_k: float = 5000.0,
    limits: CamSpringNetworkLimits | None = None,
    penalty_scale: float = 1000.0,
) -> CamSpringEnergyOptimizationResult:
    """Fit spring constants to minimize motor energy plus cam actuator energy."""

    theta_values = np.asarray(theta, dtype=float)
    time_values = np.asarray(time, dtype=float)
    theta_dot_values = np.asarray(theta_dot, dtype=float)
    phi_values = np.asarray(phi, dtype=float)
    phi_dot_values = np.asarray(phi_dot, dtype=float)
    tau_required_values = np.asarray(tau_required, dtype=float)
    baseline_power = np.asarray(motor_power_without_spring, dtype=float)
    if not (
        time_values.shape
        == theta_values.shape
        == theta_dot_values.shape
        == phi_values.shape
        == phi_dot_values.shape
        == tau_required_values.shape
        == baseline_power.shape
    ):
        raise ValueError("all trajectory arrays must have the same shape")

    if initial_k is None:
        initial = np.array([spring.k for spring in template_network.springs], dtype=float)
    else:
        initial = np.asarray(initial_k, dtype=float)
    if initial.size != len(template_network.springs):
        raise ValueError("initial_k must have one value per spring")

    def objective(k_values: np.ndarray) -> float:
        network = _network_with_spring_rates(template_network, k_values)
        tau_spring = network.torque(theta_values, phi_values)
        motor_power_with = residual_motor_power(tau_required_values, tau_spring, theta_dot_values)
        cam_power = positive_cam_power(network.cam_torque(theta_values, phi_values), phi_dot_values)
        energy = float(np.trapezoid(motor_power_with + cam_power, time_values))
        if limits is not None:
            report = network.check_constraints(theta_values, phi_values, phi_dot=phi_dot_values, limits=limits)
            energy += penalty_scale * _constraint_penalty(report)
        return energy

    result = minimize(
        objective,
        np.clip(initial, min_k, max_k),
        method="L-BFGS-B",
        bounds=[(min_k, max_k)] * initial.size,
    )
    network = _network_with_spring_rates(template_network, result.x)
    tau_spring = network.torque(theta_values, phi_values)
    motor_power_with = residual_motor_power(tau_required_values, tau_spring, theta_dot_values)
    cam_power = positive_cam_power(network.cam_torque(theta_values, phi_values), phi_dot_values)
    motor_energy = float(np.trapezoid(motor_power_with, time_values))
    cam_energy = float(np.trapezoid(cam_power, time_values))
    baseline_energy = float(np.trapezoid(baseline_power, time_values))
    net_saved = baseline_energy - motor_energy - cam_energy
    net_offload = 0.0 if baseline_energy == 0.0 else net_saved / baseline_energy * 100.0
    return CamSpringEnergyOptimizationResult(
        network=network,
        optimized_spring_rates=np.asarray(result.x, dtype=float),
        objective_energy=float(result.fun),
        motor_energy=motor_energy,
        cam_actuator_energy=cam_energy,
        net_energy_saved=net_saved,
        net_offload_percentage=net_offload,
        success=bool(result.success),
        message=str(result.message),
    )


def smooth_engagement(
    phi: float | np.ndarray,
    phi_start: float,
    phi_end: float,
) -> np.ndarray:
    """
    Return clipped linear spring engagement from 0 to 1.

    0 means not engaged, 1 means fully engaged.
    """

    if phi_end <= phi_start:
        raise ValueError("phi_end must be greater than phi_start")
    s = (np.asarray(phi, dtype=float) - phi_start) / (phi_end - phi_start)
    return np.clip(s, 0.0, 1.0)


def default_three_spring_network() -> CamSpringNetwork:
    return CamSpringNetwork(
        springs=[
            CamSpring(
                k=500.0,
                preload=0.002,
                lever_arm=0.04,
                phi_start=0.0,
                phi_end=0.4,
                max_compression=0.004,
            ),
            CamSpring(
                k=1000.0,
                preload=0.0,
                lever_arm=0.04,
                phi_start=0.3,
                phi_end=0.8,
                max_compression=0.006,
            ),
            CamSpring(
                k=1800.0,
                preload=0.0,
                lever_arm=0.04,
                phi_start=0.7,
                phi_end=1.2,
                max_compression=0.008,
            ),
        ],
        theta_rest=0.0,
        assist_direction=1.0,
    )


def positive_cam_power(cam_torque: np.ndarray, phi_dot: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(cam_torque, dtype=float) * np.asarray(phi_dot, dtype=float))


def _torque_basis_for_spring_rates(network: CamSpringNetwork, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    columns = []
    for spring in network.springs:
        unit_spring = CamSpring(
            k=1.0,
            preload=spring.preload,
            lever_arm=spring.lever_arm,
            phi_start=spring.phi_start,
            phi_end=spring.phi_end,
            max_compression=spring.max_compression,
        )
        unit_network = CamSpringNetwork(
            [unit_spring],
            theta_rest=network.theta_rest,
            assist_direction=network.assist_direction,
        )
        columns.append(unit_network.torque(theta, phi))
    return np.column_stack(columns)


def _network_with_spring_rates(template_network: CamSpringNetwork, spring_rates: np.ndarray) -> CamSpringNetwork:
    return CamSpringNetwork(
        [
            CamSpring(
                k=float(k),
                preload=spring.preload,
                lever_arm=spring.lever_arm,
                phi_start=spring.phi_start,
                phi_end=spring.phi_end,
                max_compression=spring.max_compression,
            )
            for k, spring in zip(spring_rates, template_network.springs)
        ],
        theta_rest=template_network.theta_rest,
        assist_direction=template_network.assist_direction,
    )


def _constraint_penalty(report: CamSpringNetworkConstraintReport) -> float:
    terms = [
        _normalized_excess(report.max_compression, report.limits.max_compression),
        _normalized_excess(report.max_force, report.limits.max_force),
        _normalized_excess(report.max_abs_spring_torque, report.limits.max_abs_spring_torque),
        _normalized_excess(report.max_abs_cam_torque, report.limits.max_abs_cam_torque),
        _normalized_excess(report.max_abs_phi_speed, report.limits.max_abs_phi_speed),
    ]
    return float(np.sum(np.square(terms)))


def _normalized_excess(value: float, limit: float) -> float:
    if limit <= 0.0:
        return 0.0
    return max(0.0, (value - limit) / limit)


def _exceeds_limit(value: float, limit: float) -> bool:
    return value > limit * (1.0 + 1e-6) + 1e-12


def _expand_spring_axis(values: np.ndarray, ndim: int) -> np.ndarray:
    return values.reshape((values.size,) + (1,) * (ndim - 1))
