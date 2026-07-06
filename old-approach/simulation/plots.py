from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_state(results: dict[str, object]):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True)
    for label, result in results.items():
        axes[0].plot(result.time, result.theta, label=label)
        axes[1].plot(result.time, result.theta_dot, label=label)
    axes[0].set_ylabel("theta (rad)")
    axes[1].set_ylabel("theta_dot (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    return fig, axes


def plot_adaptive_state(results: dict[str, object]):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True)
    for label, result in results.items():
        axes[0].plot(result.time, result.q, label=label)
        axes[1].plot(result.time, result.k_eff, label=label)
    axes[0].set_ylabel("q")
    axes[1].set_ylabel("k_eff (N m/rad)")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    return fig, axes


def plot_torques_and_power(results: dict[str, object]):
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True, constrained_layout=True)
    for label, result in results.items():
        axes[0].plot(result.time, result.tau_spring, label=label)
        axes[1].plot(result.time, result.tau_motor, label=label)
        axes[2].plot(result.time, result.motor_power, label=label)
    axes[0].set_ylabel("spring torque (N m)")
    axes[1].set_ylabel("motor torque (N m)")
    axes[2].set_ylabel("motor power (W)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend()
    return fig, axes


def plot_energy_comparison(energy_summaries: dict[str, object]):
    labels = list(energy_summaries)
    average_power = [energy_summaries[label].average_motor_power for label in labels]
    offload = [energy_summaries[label].offload_percentage for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].bar(labels, average_power)
    axes[0].set_ylabel("average motor power (W)")
    axes[1].bar(labels, offload)
    axes[1].set_ylabel("offload (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    return fig, axes


def save_all_plots(results: dict[str, object], energy_summaries: dict[str, object], output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figures = [
        ("joint_state.png", plot_joint_state(results)[0]),
        ("adaptive_state.png", plot_adaptive_state(results)[0]),
        ("torques_and_power.png", plot_torques_and_power(results)[0]),
        ("energy_comparison.png", plot_energy_comparison(energy_summaries)[0]),
    ]
    paths = []
    for filename, fig in figures:
        path = output_path / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
