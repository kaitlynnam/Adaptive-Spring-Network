from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from python.pej.core import motor_power, offload_percentage


@dataclass(frozen=True)
class EnergySummary:
    total_motor_energy: float
    average_motor_power: float
    peak_motor_power: float
    offload_percentage: float


@dataclass(frozen=True)
class NetEnergySummary:
    motor_energy_without_spring: float
    motor_energy_with_spring: float
    cam_actuator_energy: float
    motor_energy_saved: float
    net_energy_saved: float
    gross_offload_percentage: float
    net_offload_percentage: float


def summarize_energy(
    time: np.ndarray,
    tau_motor: np.ndarray,
    theta_dot: np.ndarray,
    *,
    baseline_power: np.ndarray | None = None,
) -> EnergySummary:
    """Compute motor energy and power metrics for one simulation."""

    time_values = np.asarray(time, dtype=float)
    power = motor_power(tau_motor, theta_dot)
    total_energy = float(np.trapezoid(power, time_values))
    duration = float(time_values[-1] - time_values[0])
    if duration <= 0.0:
        raise ValueError("time must span a positive duration")
    average_power = total_energy / duration
    peak_power = float(np.max(power))
    if baseline_power is None:
        offload = 0.0
    else:
        baseline_average = float(np.trapezoid(np.asarray(baseline_power, dtype=float), time_values) / duration)
        offload = float(offload_percentage(baseline_average, average_power))
    return EnergySummary(
        total_motor_energy=total_energy,
        average_motor_power=average_power,
        peak_motor_power=peak_power,
        offload_percentage=offload,
    )


def compare_energy(results: dict[str, object]) -> dict[str, EnergySummary]:
    """Summarize multiple simulation results using the first no-spring result as baseline if present."""

    baseline = results.get("no_spring")
    baseline_power = None if baseline is None else baseline.motor_power
    return {
        name: summarize_energy(result.time, result.tau_motor, result.theta_dot, baseline_power=baseline_power)
        for name, result in results.items()
    }


def net_energy_savings(
    time: np.ndarray,
    motor_power_without_spring: np.ndarray,
    motor_power_with_spring: np.ndarray,
    cam_actuator_power: np.ndarray,
) -> NetEnergySummary:
    """Compute motor savings minus positive cam actuator energy."""

    time_values = np.asarray(time, dtype=float)
    duration = float(time_values[-1] - time_values[0])
    if duration <= 0.0:
        raise ValueError("time must span a positive duration")
    motor_energy_without = float(np.trapezoid(np.asarray(motor_power_without_spring, dtype=float), time_values))
    motor_energy_with = float(np.trapezoid(np.asarray(motor_power_with_spring, dtype=float), time_values))
    cam_energy = float(np.trapezoid(np.asarray(cam_actuator_power, dtype=float), time_values))
    motor_saved = motor_energy_without - motor_energy_with
    net_saved = motor_saved - cam_energy
    gross_offload = 0.0 if motor_energy_without == 0.0 else motor_saved / motor_energy_without * 100.0
    net_offload = 0.0 if motor_energy_without == 0.0 else net_saved / motor_energy_without * 100.0
    return NetEnergySummary(
        motor_energy_without_spring=motor_energy_without,
        motor_energy_with_spring=motor_energy_with,
        cam_actuator_energy=cam_energy,
        motor_energy_saved=motor_saved,
        net_energy_saved=net_saved,
        gross_offload_percentage=gross_offload,
        net_offload_percentage=net_offload,
    )
