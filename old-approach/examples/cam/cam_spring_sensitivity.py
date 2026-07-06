#!/usr/bin/env python3
"""One-at-a-time sensitivity analysis for the cam spring network demo."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from examples.common.reporting import print_section, print_table, print_written, write_csv
from python.pej.core import motor_power, residual_motor_power
from cam_spring_network import prescribed_theta, required_joint_torque, roughness_schedule
from simulation.actuator import CamActuatorConfig, update_cam_actuator
from simulation.cam_spring_network import (
    CamSpring,
    CamSpringNetwork,
    CamSpringNetworkLimits,
    default_three_spring_network,
    fit_spring_rates_to_targets,
    optimize_spring_rates_for_energy,
    positive_cam_power,
)
from simulation.energy import net_energy_savings


@dataclass(frozen=True)
class Scenario:
    parameter: str
    multiplier: float
    spring_rates: np.ndarray
    phi_scale: float
    tau_response: float
    max_speed: float


def main() -> None:
    context = build_baseline()
    scenarios = build_scenarios(context["spring_rates"])
    rows = [evaluate_scenario(context, scenario) for scenario in scenarios]
    baseline_offload = float(next(row for row in rows if row["scenario"] == "baseline")["offload_pct"])
    for row in rows:
        row["delta_pct_point"] = float(row["offload_pct"]) - baseline_offload

    output_dir = Path("artifacts/plots/cam_spring_network")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(
        "artifacts/tables/cam_spring_sensitivity.csv",
        rows,
        [
            "scenario",
            "case",
            "offload_pct",
            "delta_pct_point",
            "net_saved_j",
            "cam_energy_j",
            "spring_k_1",
            "spring_k_2",
            "spring_k_3",
            "constraints_passed",
        ],
    )
    plot_path = write_plot(output_dir / "sensitivity_net_offload.png", rows)

    baseline = next(row for row in rows if row["scenario"] == "baseline")
    print("Cam spring network sensitivity analysis")
    print(f"  Baseline spring rates: {', '.join(f'{k:.1f}' for k in context['spring_rates'])} N/m")
    print(f"  Baseline net offload:  {float(baseline['offload_pct']):.2f} %")
    print_section("Sensitivity comparison")
    print_table(
        rows,
        [
            "scenario",
            "case",
            "offload_pct",
            "delta_pct_point",
            "net_saved_j",
            "cam_energy_j",
            "spring_k_1",
            "spring_k_2",
            "spring_k_3",
            "constraints_passed",
        ],
    )
    print_written([csv_path], "Wrote table")
    print_written([plot_path], "Wrote plot")


def build_baseline() -> dict[str, np.ndarray | float]:
    time = np.arange(0.0, 10.0 + 0.01, 0.01)
    theta = prescribed_theta(time)
    theta_dot = np.gradient(theta, time)
    q = roughness_schedule(time)
    template = default_three_spring_network()
    phi, phi_dot = simulate_cam(time, 1.2 * q, tau_response=0.18, max_speed=3.0)
    tau_required = required_joint_torque(theta, theta_dot, q)
    baseline_power = motor_power(tau_required, theta_dot)
    tau_abstract = -(0.9 + q * 2.2) * (theta - template.theta_rest)
    fit = fit_spring_rates_to_targets(template, theta, phi, target_torque=tau_abstract, min_k=0.0, max_k=5000.0)
    limits = CamSpringNetworkLimits(
        max_compression=0.030,
        max_force=80.0,
        max_abs_spring_torque=8.0,
        max_abs_cam_torque=0.8,
        max_abs_phi_speed=3.1,
    )
    opt = optimize_spring_rates_for_energy(
        template,
        time,
        theta,
        theta_dot,
        phi,
        phi_dot,
        tau_required,
        baseline_power,
        initial_k=fit.fitted_spring_rates,
        min_k=0.0,
        max_k=5000.0,
        limits=limits,
        penalty_scale=1e6,
    )
    return {
        "time": time,
        "theta": theta,
        "theta_dot": theta_dot,
        "q": q,
        "template": template,
        "tau_required": tau_required,
        "baseline_power": baseline_power,
        "spring_rates": opt.optimized_spring_rates,
        "limits": limits,
    }


def build_scenarios(spring_rates: np.ndarray) -> list[Scenario]:
    scenarios = [
        Scenario("baseline", 1.0, spring_rates.copy(), 1.0, 0.18, 3.0),
    ]
    for i in range(spring_rates.size):
        for multiplier in (0.8, 1.2):
            varied = spring_rates.copy()
            varied[i] *= multiplier
            scenarios.append(Scenario(f"k_spring_{i + 1}", multiplier, varied, 1.0, 0.18, 3.0))
    for parameter, attr in (("phi_scale", "phi_scale"), ("tau_response", "tau_response"), ("max_speed", "max_speed")):
        for multiplier in (0.8, 1.2):
            values = {"phi_scale": 1.0, "tau_response": 0.18, "max_speed": 3.0}
            values[attr] *= multiplier
            scenarios.append(
                Scenario(parameter, multiplier, spring_rates.copy(), values["phi_scale"], values["tau_response"], values["max_speed"])
            )
    return scenarios


def evaluate_scenario(context: dict[str, object], scenario: Scenario) -> dict[str, str]:
    time = context["time"]
    theta = context["theta"]
    theta_dot = context["theta_dot"]
    q = context["q"]
    tau_required = context["tau_required"]
    baseline_power = context["baseline_power"]
    phi, phi_dot = simulate_cam(time, 1.2 * scenario.phi_scale * q, scenario.tau_response, scenario.max_speed)
    network = network_with_rates(context["template"], scenario.spring_rates)
    tau_spring = network.torque(theta, phi)
    motor_power_with = residual_motor_power(tau_required, tau_spring, theta_dot)
    cam_power = positive_cam_power(network.cam_torque(theta, phi), phi_dot)
    net = net_energy_savings(time, baseline_power, motor_power_with, cam_power)
    report = network.check_constraints(theta, phi, phi_dot=phi_dot, limits=context["limits"])
    return {
        "scenario": scenario.parameter,
        "case": f"x{scenario.multiplier:.2f}",
        "offload_pct": net.net_offload_percentage,
        "delta_pct_point": "",
        "net_saved_j": net.net_energy_saved,
        "motor_energy_j": net.motor_energy_with_spring,
        "cam_energy_j": net.cam_actuator_energy,
        "spring_k_1": scenario.spring_rates[0],
        "spring_k_2": scenario.spring_rates[1],
        "spring_k_3": scenario.spring_rates[2],
        "constraints_passed": str(report.passed),
        "violations": ",".join(report.violations),
    }


def simulate_cam(time: np.ndarray, phi_desired: np.ndarray, tau_response: float, max_speed: float) -> tuple[np.ndarray, np.ndarray]:
    config = CamActuatorConfig(tau_response=tau_response, max_speed=max_speed)
    phi = np.empty_like(time)
    phi_dot = np.empty_like(time)
    phi[0] = phi_desired[0]
    phi_dot[0] = 0.0
    for i in range(1, time.size):
        step = update_cam_actuator(phi[i - 1], phi_desired[i], float(time[i] - time[i - 1]), config)
        phi[i] = step.phi
        phi_dot[i] = step.phi_dot
    return phi, phi_dot


def network_with_rates(template: CamSpringNetwork, spring_rates: np.ndarray) -> CamSpringNetwork:
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
            for k, spring in zip(spring_rates, template.springs)
        ],
        theta_rest=template.theta_rest,
        assist_direction=template.assist_direction,
    )


def write_plot(path: Path, rows: list[dict[str, object]]) -> Path:
    labels = [f"{row['scenario']} {row['case']}" for row in rows]
    values = [float(row["offload_pct"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    ax.bar(labels, values)
    ax.set_ylabel("net offload (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
