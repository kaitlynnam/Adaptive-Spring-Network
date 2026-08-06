"""Deploy a trained period-adaptive controller on a sequence of trajectories."""

from pathlib import Path
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES
from benchmark_profile_passive_3d import spatial_initial_basis
from mechanics_3d import load_spatial_topology
from profile_generator import generate_profile_parameters
from train_period_adaptive_3d import (
    DEFAULT_TOPOLOGY, build_period_dataset, exact_period_torque,
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "period_adaptive_3d" / "period_adaptive_3d_60spring.npz"
)


def load_checkpoint(path):
    saved = np.load(path, allow_pickle=False)
    model = {name: saved[name] for name in ("w1", "b1", "w2", "b2")}
    metadata = {name: saved[name] for name in saved.files if name not in model}
    return model, metadata


def softplus(values):
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def encode_measured_period(dataset, index, spring_torque, scales, torque_scale):
    channels = np.column_stack((
        dataset["theta"][index] / scales[0],
        dataset["theta_dot"][index] / scales[1],
        dataset["theta_ddot"][index] / scales[2],
        dataset["target"][index] / torque_scale,
        spring_torque / torque_scale,
        (dataset["target"][index] - spring_torque) / torque_scale,
    ))
    return channels.reshape(1, -1)


def controller_step(model, observation, min_stiffness,
                    stiffness_lower=None, stiffness_upper=None):
    hidden = np.tanh(observation @ model["w1"] + model["b1"])
    logits = hidden @ model["w2"] + model["b2"]
    if stiffness_lower is not None and stiffness_upper is not None:
        stiffness = np.clip(
            min_stiffness + softplus(logits), stiffness_lower, stiffness_upper
        )
    else:
        stiffness = min_stiffness + softplus(logits)
    return stiffness[0] if len(stiffness) == 1 else stiffness


def dataset_period(dataset, index):
    result = dict(dataset)
    for key in ("theta", "theta_dot", "theta_ddot", "target", "basis", "t"):
        result[key] = dataset[key][index:index + 1]
    return result


def deploy(model, metadata, dataset, topology, relaxation_steps, batch_size,
           progress_interval):
    """Run the stateful controller, including its no-input first period."""
    period_count = len(dataset["target"])
    initial_k = np.asarray(metadata["initial_stiffness"], dtype=float)
    stiffness = initial_k.copy()
    stiffness_schedule, torque_rows, residual_rows = [], [], []
    scales = np.asarray(metadata["motion_scales"], dtype=float)
    torque_scale = float(metadata["torque_scale"])
    min_k = float(metadata["min_k"])
    lower = metadata.get("stiffness_lower_bound")
    upper = metadata.get("stiffness_upper_bound")
    for index in range(period_count):
        stiffness_schedule.append(stiffness.copy())
        torque, residual = exact_period_torque(
            dataset_period(dataset, index), topology, stiffness[None, None, :],
            relaxation_steps, batch_size, progress_interval,
        )
        measured_torque = torque[0, 0]
        torque_rows.append(measured_torque)
        residual_rows.append(residual[0, 0])
        observation = encode_measured_period(
            dataset, index, measured_torque, scales, torque_scale
        )
        stiffness = controller_step(model, observation, min_k, lower, upper)
    return (np.asarray(torque_rows), np.asarray(stiffness_schedule),
            np.asarray(residual_rows))


def save_figures(output_dir, name, dataset, torque, stiffness):
    output_dir.mkdir(parents=True, exist_ok=True)
    periods = len(torque)
    period_seconds = dataset["period_seconds"]
    colors = ["#888888"] + [plt.cm.viridis(x) for x in np.linspace(0.25, 0.9, max(periods - 1, 1))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for index in range(periods):
        time = dataset["t"][index] - dataset["t"][index, 0] + index * period_seconds
        ax.plot(time, dataset["target"][index], "k--", alpha=0.7,
                label="target" if index == 0 else None)
        ax.plot(time, torque[index], color=colors[index], linewidth=2,
                label=f"spring period {index + 1}{' (default)' if index == 0 else ''}")
        if index:
            ax.axvline(index * period_seconds, color="0.8", linewidth=0.8)
    ax.set(xlabel="Time [s]", ylabel="Torque [N m]",
           title="Deployed period-adaptive torque")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_deployment_torque_time.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(periods, 1, figsize=(7.5, max(3.2, 2.8 * periods)), squeeze=False)
    for index, ax in enumerate(axes[:, 0]):
        angle = np.degrees(dataset["theta"][index])
        ax.plot(angle, dataset["target"][index], "k--", linewidth=2, label="target")
        ax.plot(angle, torque[index], color=colors[index], linewidth=2,
                label="spring (default)" if index == 0 else "spring (from previous period)")
        ax.set(xlabel="Joint angle [deg]", ylabel="Torque [N m]", title=f"Period {index + 1}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_deployment_torque_angle.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.55 * periods)))
    image = ax.imshow(stiffness, aspect="auto", cmap="viridis")
    ax.set(xlabel="Spring index", ylabel="Period", title="Deployed stiffness schedule")
    ax.set_yticks(np.arange(periods), [f"{i + 1}{' default' if i == 0 else ''}" for i in range(periods)])
    fig.colorbar(image, ax=ax, label="Stiffness [N/m]")
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_deployment_stiffness.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--trajectory-mode", choices=["repeated", "changing"], default="repeated")
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--mechanics-batch-size", type=int, default=1024)
    parser.add_argument("--mechanics-progress-interval", type=int, default=10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--output-name", default="period_adaptive_3d_60spring")
    args = parser.parse_args()
    if args.periods < 1:
        parser.error("--periods must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model, metadata = load_checkpoint(args.checkpoint)
    samples = int(metadata["samples_per_period"])
    period_seconds = float(metadata["period_seconds"])
    topology = load_spatial_topology(args.topology, torch.device(args.device))
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, args.relaxation_steps)
    rng = np.random.default_rng(args.seed)
    profiles = generate_profile_parameters(rng, 1 if args.trajectory_mode == "repeated" else args.periods)
    if args.trajectory_mode == "repeated":
        profiles = [dict(profiles[0]) for _ in range(args.periods)]
    dataset = build_period_dataset(
        profiles, angles, basis, period_seconds, samples, args.seed + 10_000,
        motion_mode=str(metadata["motion_mode"]), frequency_hz=1.0 / period_seconds,
        torque_scale=float(metadata["torque_scale"]),
    )
    torque, stiffness, force_residual = deploy(
        model, metadata, dataset, topology, args.relaxation_steps,
        args.mechanics_batch_size, args.mechanics_progress_interval,
    )
    rmse = np.sqrt(np.mean((torque - dataset["target"]) ** 2, axis=1))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    baseline_work = np.asarray([
        trapezoid(np.abs(dataset["target"][i] * dataset["theta_dot"][i]), dataset["t"][i])
        for i in range(args.periods)
    ])
    assisted_work = np.asarray([
        trapezoid(np.abs((dataset["target"][i] - torque[i]) * dataset["theta_dot"][i]),
                  dataset["t"][i])
        for i in range(args.periods)
    ])
    offload = 100.0 * (baseline_work - assisted_work) / np.maximum(baseline_work, 1e-12)
    for index, value in enumerate(rmse):
        policy = "default" if index == 0 else "previous-period update"
        print(f"Period {index + 1} ({policy}): RMSE {value:.3f} N*m | "
              f"motor-work offload {offload[index]:.3f}%")
    adapted_aggregate = 100.0 * (
        1.0 - np.sum(assisted_work[1:]) / np.maximum(np.sum(baseline_work[1:]), 1e-12)
    ) if args.periods > 1 else float("nan")
    print(f"Adapted-period aggregate motor-work offload: {adapted_aggregate:.3f}%")
    output_dir = PROJECT_ROOT / "plots" / "period_adaptive_3d"
    save_figures(output_dir, args.output_name, dataset, torque, stiffness)
    table = PROJECT_ROOT / "tables" / "period_adaptive_3d" / f"{args.output_name}_deployment.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "period", "policy", "rmse_nm", "motor_work_offload_pct",
            "baseline_motor_work_j", "assisted_motor_work_j", "mean_force_residual_n",
        ])
        writer.writeheader()
        for index, value in enumerate(rmse):
            writer.writerow({"period": index + 1, "policy": "default" if index == 0 else "previous_period",
                             "rmse_nm": value, "motor_work_offload_pct": offload[index],
                             "baseline_motor_work_j": baseline_work[index],
                             "assisted_motor_work_j": assisted_work[index],
                             "mean_force_residual_n": np.mean(force_residual[index])})
    print(f"Saved deployment figures to {output_dir}")
    print(f"Saved deployment metrics to {table}")


if __name__ == "__main__":
    main()
