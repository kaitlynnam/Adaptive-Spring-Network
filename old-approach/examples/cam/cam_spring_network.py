#!/usr/bin/env python3
"""Demo a cam-controlled 3-spring adaptive PEJ network."""

from __future__ import annotations

import os
import sys
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
from simulation.actuator import CamActuatorConfig, update_cam_actuator
from simulation.cam_spring_network import (
    CamSpringNetworkLimits,
    default_three_spring_network,
    fit_spring_rates_to_targets,
    optimize_spring_rates_for_energy,
    positive_cam_power,
)
from simulation.energy import net_energy_savings


def main() -> None:
    time = np.arange(0.0, 10.0 + 0.01, 0.01)
    theta = prescribed_theta(time)
    theta_dot = np.gradient(theta, time)
    q = roughness_schedule(time)
    phi_desired = 1.2 * q
    phi, phi_dot = simulate_cam_actuator(time, phi_desired)

    template_network = default_three_spring_network()
    tau_required = required_joint_torque(theta, theta_dot, q)
    tau_fixed_soft = -0.9 * (theta - template_network.theta_rest)
    tau_abstract = -(0.9 + q * 2.2) * (theta - template_network.theta_rest)
    fit = fit_spring_rates_to_targets(
        template_network,
        theta,
        phi,
        target_torque=tau_abstract,
        min_k=0.0,
        max_k=5000.0,
    )
    limits = CamSpringNetworkLimits(
        max_compression=0.030,
        max_force=80.0,
        max_abs_spring_torque=8.0,
        max_abs_cam_torque=0.8,
        max_abs_phi_speed=3.1,
    )
    energy_opt = optimize_spring_rates_for_energy(
        template_network,
        time,
        theta,
        theta_dot,
        phi,
        phi_dot,
        tau_required,
        motor_power(tau_required, theta_dot),
        initial_k=fit.fitted_spring_rates,
        min_k=0.0,
        max_k=5000.0,
        limits=limits,
        penalty_scale=1e6,
    )
    network = energy_opt.network

    tau_cam_spring = network.torque(theta, phi)
    compressions = network.spring_compressions(theta, phi)
    forces = network.spring_forces(theta, phi)
    cam_torque = network.cam_torque(theta, phi)
    cam_power = positive_cam_power(cam_torque, phi_dot)

    cases = {
        "no spring": np.zeros_like(theta),
        "fixed soft": tau_fixed_soft,
        "abstract adaptive": tau_abstract,
        "cam network": tau_cam_spring,
    }
    motor_powers = {
        name: residual_motor_power(tau_required, tau_spring, theta_dot)
        for name, tau_spring in cases.items()
    }
    baseline_power = motor_power(tau_required, theta_dot)
    net = net_energy_savings(time, baseline_power, motor_powers["cam network"], cam_power)
    constraints = network.check_constraints(theta, phi, phi_dot=phi_dot, limits=limits)
    energy_error = network.cam_torque_energy_error(theta=0.30, phi=0.55)

    output_dir = Path("artifacts/plots/cam_spring_network")
    output_dir.mkdir(parents=True, exist_ok=True)
    spring_rates = energy_opt.optimized_spring_rates
    table_rows = comparison_table(time, baseline_power, motor_powers, cam_power, spring_rates)
    table_path = write_csv("artifacts/tables/cam_spring_network_energy.csv", table_rows)
    segment_rows = terrain_segment_rows(time, q, baseline_power, motor_powers["cam network"], cam_power, spring_rates)
    segment_table_path = write_csv("artifacts/tables/cam_spring_network_segments.csv", segment_rows)
    plot_paths = write_plots(
        output_dir,
        time,
        theta,
        theta_dot,
        q,
        phi,
        compressions,
        forces,
        cases,
        motor_powers,
        baseline_power,
        cam_power,
    )

    print("Cam-controlled 3-spring adaptive PEJ demo")
    print(f"  Fit-start spring rates:      {', '.join(f'{k:.1f}' for k in fit.fitted_spring_rates)} N/m")
    print(f"  Energy-opt spring rates:     {', '.join(f'{k:.1f}' for k in energy_opt.optimized_spring_rates)} N/m")
    print(f"  Energy optimizer:            {'success' if energy_opt.success else 'failed'}")
    print(f"  Fit RMS / max error:         {fit.rms_error:.4f} / {fit.max_abs_error:.4f} N m")
    print(f"  Cam torque energy error:     {energy_error:.6f} N m")
    print(f"  Constraint check:            {'passed' if constraints.passed else 'failed'}")
    if constraints.violations:
        print(f"  Constraint violations:       {', '.join(constraints.violations)}")
    print(f"  Motor energy without spring: {net.motor_energy_without_spring:.4f} J")
    print(f"  Motor energy with cam PEJ:   {net.motor_energy_with_spring:.4f} J")
    print(f"  Cam actuator energy:         {net.cam_actuator_energy:.4f} J")
    print(f"  Motor energy saved:          {net.motor_energy_saved:.4f} J")
    print(f"  Net energy saved:            {net.net_energy_saved:.4f} J")
    print(f"  Gross offload:               {net.gross_offload_percentage:.2f} %")
    print(f"  Net offload:                 {net.net_offload_percentage:.2f} %")
    print_section("Energy comparison")
    print_table(
        table_rows,
        [
            "scenario",
            "case",
            "motor_energy_j",
            "cam_energy_j",
            "net_saved_j",
            "offload_pct",
            "spring_k_1",
            "spring_k_2",
            "spring_k_3",
            "peak_motor_power_w",
        ],
    )
    print_section("Terrain segments")
    print_table(
        segment_rows,
        [
            "scenario",
            "case",
            "mean_q",
            "baseline_motor_energy_j",
            "motor_energy_j",
            "cam_energy_j",
            "net_saved_j",
            "offload_pct",
            "spring_k_1",
            "spring_k_2",
            "spring_k_3",
        ],
    )
    print_written([table_path, segment_table_path], "Wrote table")
    print_written(plot_paths, "Wrote plot")


def prescribed_theta(time: np.ndarray) -> np.ndarray:
    return 0.28 + 0.12 * np.sin(2.0 * np.pi * 1.2 * time) + 0.03 * np.sin(2.0 * np.pi * 3.4 * time + 0.3)


def roughness_schedule(time: np.ndarray) -> np.ndarray:
    ramp = np.clip((time - 3.0) / 2.0, 0.0, 1.0)
    settle = np.clip((time - 7.5) / 1.5, 0.0, 1.0)
    return np.clip(ramp * (1.0 - 0.35 * settle), 0.0, 1.0)


def required_joint_torque(theta: np.ndarray, theta_dot: np.ndarray, q: np.ndarray) -> np.ndarray:
    terrain_stiffness = 3.0 + 8.0 * q
    terrain_damping = 0.04 + 0.08 * q
    return -(terrain_stiffness * theta + terrain_damping * theta_dot)


def simulate_cam_actuator(time: np.ndarray, phi_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    config = CamActuatorConfig(tau_response=0.18, max_speed=3.0)
    phi = np.empty_like(time)
    phi_dot = np.empty_like(time)
    phi[0] = phi_desired[0]
    phi_dot[0] = 0.0
    for i in range(1, time.size):
        dt = float(time[i] - time[i - 1])
        step = update_cam_actuator(phi[i - 1], phi_desired[i], dt, config)
        phi[i] = step.phi
        phi_dot[i] = step.phi_dot
    return phi, phi_dot


def write_plots(
    output_dir: Path,
    time: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    q: np.ndarray,
    phi: np.ndarray,
    compressions: np.ndarray,
    forces: np.ndarray,
    cases: dict[str, np.ndarray],
    motor_powers: dict[str, np.ndarray],
    baseline_power: np.ndarray,
    cam_power: np.ndarray,
) -> list[Path]:
    paths = [
        save_plot(output_dir / "joint_angle_velocity.png", lambda ax: plot_joint(ax, time, theta, theta_dot)),
        save_plot(output_dir / "cam_angle.png", lambda ax: plot_lines(ax, time, {"q": q, "phi": phi}, "q / phi")),
        save_plot(output_dir / "spring_compressions.png", lambda ax: plot_spring_rows(ax, time, compressions, "compression (m)")),
        save_plot(output_dir / "spring_forces.png", lambda ax: plot_spring_rows(ax, time, forces, "force (N)")),
        save_plot(output_dir / "spring_torque.png", lambda ax: plot_lines(ax, time, cases, "spring torque (N m)")),
        save_plot(
            output_dir / "motor_and_cam_power.png",
            lambda ax: plot_lines(ax, time, {"no spring": baseline_power, **motor_powers, "cam actuator": cam_power}, "power (W)"),
        ),
        save_plot(
            output_dir / "cumulative_energy.png",
            lambda ax: plot_cumulative_energy(ax, time, baseline_power, motor_powers, cam_power),
        ),
    ]
    return paths


def comparison_table(
    time: np.ndarray,
    baseline_power: np.ndarray,
    motor_powers: dict[str, np.ndarray],
    cam_power: np.ndarray,
    spring_rates: np.ndarray,
) -> list[dict[str, str]]:
    baseline_energy = float(np.trapezoid(baseline_power, time))
    rows = []
    for name, power in {"no spring": baseline_power, **motor_powers}.items():
        motor_energy = float(np.trapezoid(power, time))
        cam_energy = float(np.trapezoid(cam_power, time)) if name == "cam network + cam cost" else 0.0
        if name == "cam network + cam cost":
            motor_energy = float(np.trapezoid(motor_powers["cam network"], time))
        net_saved = baseline_energy - motor_energy - cam_energy
        rows.append(
            {
                "scenario": "whole run",
                "case": name,
                "motor_energy_j": f"{motor_energy:.6f}",
                "cam_energy_j": f"{cam_energy:.6f}",
                "net_saved_j": f"{net_saved:.6f}",
                "offload_pct": f"{net_saved / baseline_energy * 100.0:.6f}",
                "spring_k_1": _spring_rate_cell(name, spring_rates, 0),
                "spring_k_2": _spring_rate_cell(name, spring_rates, 1),
                "spring_k_3": _spring_rate_cell(name, spring_rates, 2),
                "peak_motor_power_w": f"{float(np.max(power)):.6f}",
            }
        )
    rows.append(
        {
            "scenario": "whole run",
            "case": "cam network + cam cost",
            "motor_energy_j": f"{float(np.trapezoid(motor_powers['cam network'], time)):.6f}",
            "cam_energy_j": f"{float(np.trapezoid(cam_power, time)):.6f}",
            "net_saved_j": f"{baseline_energy - float(np.trapezoid(motor_powers['cam network'], time)) - float(np.trapezoid(cam_power, time)):.6f}",
            "offload_pct": f"{(baseline_energy - float(np.trapezoid(motor_powers['cam network'], time)) - float(np.trapezoid(cam_power, time))) / baseline_energy * 100.0:.6f}",
            "spring_k_1": f"{spring_rates[0]:.6f}",
            "spring_k_2": f"{spring_rates[1]:.6f}",
            "spring_k_3": f"{spring_rates[2]:.6f}",
            "peak_motor_power_w": f"{float(np.max(motor_powers['cam network'])):.6f}",
        }
    )
    return rows


def terrain_segment_rows(
    time: np.ndarray,
    q: np.ndarray,
    baseline_power: np.ndarray,
    cam_motor_power: np.ndarray,
    cam_power: np.ndarray,
    spring_rates: np.ndarray,
) -> list[dict[str, object]]:
    segments = [
        ("flat / low q", time < 3.0),
        ("transition", (time >= 3.0) & (time < 5.0)),
        ("rough / high q", (time >= 5.0) & (time < 7.5)),
        ("settling mixed", time >= 7.5),
        ("whole run", np.ones_like(time, dtype=bool)),
    ]
    rows = []
    for name, mask in segments:
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        sl = slice(idx[0], idx[-1] + 1)
        summary = net_energy_savings(time[sl], baseline_power[sl], cam_motor_power[sl], cam_power[sl])
        rows.append(
            {
                "scenario": name,
                "case": "cam network + cam cost",
                "mean_q": float(np.mean(q[sl])),
                "baseline_motor_energy_j": summary.motor_energy_without_spring,
                "motor_energy_j": summary.motor_energy_with_spring,
                "cam_energy_j": summary.cam_actuator_energy,
                "net_saved_j": summary.net_energy_saved,
                "offload_pct": summary.net_offload_percentage,
                "spring_k_1": spring_rates[0],
                "spring_k_2": spring_rates[1],
                "spring_k_3": spring_rates[2],
            }
        )
    return rows


def _spring_rate_cell(case_name: str, spring_rates: np.ndarray, index: int) -> str:
    return f"{spring_rates[index]:.6f}" if case_name.startswith("cam network") else ""

def save_plot(path: Path, draw) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    draw(ax)
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_joint(ax, time: np.ndarray, theta: np.ndarray, theta_dot: np.ndarray) -> None:
    ax.plot(time, theta, label="theta")
    ax.plot(time, theta_dot, label="theta_dot")
    ax.set_ylabel("joint state")


def plot_lines(ax, time: np.ndarray, values: dict[str, np.ndarray], ylabel: str) -> None:
    for label, value in values.items():
        ax.plot(time, value, label=label)
    ax.set_ylabel(ylabel)


def plot_spring_rows(ax, time: np.ndarray, values: np.ndarray, ylabel: str) -> None:
    for i, row in enumerate(values, start=1):
        ax.plot(time, row, label=f"spring {i}")
    ax.set_ylabel(ylabel)


def plot_cumulative_energy(
    ax,
    time: np.ndarray,
    baseline_power: np.ndarray,
    motor_powers: dict[str, np.ndarray],
    cam_power: np.ndarray,
) -> None:
    ax.plot(time, cumulative(time, baseline_power), label="no spring motor")
    ax.plot(time, cumulative(time, motor_powers["cam network"]), label="cam network motor")
    ax.plot(time, cumulative(time, cam_power), label="cam actuator")
    ax.plot(
        time,
        cumulative(time, baseline_power) - cumulative(time, motor_powers["cam network"]) - cumulative(time, cam_power),
        label="net saved",
    )
    ax.set_ylabel("energy (J)")


def cumulative(time: np.ndarray, power: np.ndarray) -> np.ndarray:
    dt = np.diff(time)
    avg_power = 0.5 * (power[:-1] + power[1:])
    return np.concatenate(([0.0], np.cumsum(dt * avg_power)))


if __name__ == "__main__":
    main()
