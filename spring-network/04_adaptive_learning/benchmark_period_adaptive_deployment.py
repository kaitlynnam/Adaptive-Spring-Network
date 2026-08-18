"""Benchmark causal period-adaptive deployment over many held-out profiles."""

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
from deploy_period_adaptive_3d import controller_step, load_checkpoint
from mechanics_3d import load_spatial_topology
from period_adaptive_support import figure_path, spatial_initial_basis
from profile_generator import generate_profile_parameters
from train_period_adaptive_3d import DEFAULT_TOPOLOGY, build_period_dataset, exact_period_torque

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "period_adaptive_3d" / "period_adaptive_3d_60spring_bounded_extended.npz"


def save_benchmark_figures(output_dir, name, dataset, torque, rmse, offload):
    output_dir.mkdir(parents=True, exist_ok=True)
    periods = np.arange(1, rmse.shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    mean_rmse = np.mean(rmse, axis=0)
    axes[0, 0].plot(periods, mean_rmse, marker="o", color="#355c9a", linewidth=2)
    axes[0, 0].fill_between(periods, *np.percentile(rmse, [10, 90], axis=0),
                            color="#355c9a", alpha=0.18, label="10th–90th percentile")
    axes[0, 0].set(xlabel="Deployment period", ylabel="RMSE [N m]",
                   title="RMSE convergence across profiles")
    axes[0, 0].legend(fontsize=8)

    mean_offload = np.mean(offload, axis=0)
    axes[0, 1].plot(periods, mean_offload, marker="o", color="#2a8c62", linewidth=2)
    axes[0, 1].fill_between(periods, *np.percentile(offload, [10, 90], axis=0),
                            color="#2a8c62", alpha=0.18, label="10th–90th percentile")
    axes[0, 1].axhline(0, color="0.3", linewidth=0.8)
    axes[0, 1].set(xlabel="Deployment period", ylabel="Motor-work offload [%]",
                   title="Offload convergence across profiles")
    axes[0, 1].legend(fontsize=8)

    settled_rmse, settled_offload = rmse[:, -1], offload[:, -1]
    scatter = axes[1, 0].scatter(rmse[:, 0], settled_rmse, c=settled_offload,
                                 cmap="viridis", s=32, alpha=0.85)
    limits = [0, 1.03 * max(np.max(rmse[:, 0]), np.max(settled_rmse))]
    axes[1, 0].plot(limits, limits, "k--", linewidth=1, label="no RMSE change")
    axes[1, 0].set(xlim=limits, ylim=limits, xlabel="Default-period RMSE [N m]",
                   ylabel="Settled RMSE [N m]", title="Default versus settled performance")
    axes[1, 0].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[1, 0], label="Settled offload [%]")

    axes[1, 1].hist(settled_offload, bins=15, color="#6c5b9a", edgecolor="white")
    axes[1, 1].axvline(np.median(settled_offload), color="k", linestyle="--",
                       label=f"median {np.median(settled_offload):.1f}%")
    axes[1, 1].set(xlabel="Settled motor-work offload [%]", ylabel="Profiles",
                   title="Held-out offload distribution")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path(output_dir, name, "fig05a_many_profile_benchmark.png"), dpi=200)
    plt.close(fig)

    order = np.argsort(settled_offload)
    selected = [order[0], order[len(order) // 2], order[-1]]
    labels = ["lowest offload", "median offload", "highest offload"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), squeeze=False)
    for row, (index, label) in enumerate(zip(selected, labels)):
        ax = axes[row, 0]
        angle = np.degrees(dataset["theta"][index])
        ax.plot(angle, dataset["target"][index], "k--", linewidth=2, label="target")
        ax.plot(angle, torque[index, 0], color="0.55", linewidth=2,
                label="period 1 default")
        ax.plot(angle, torque[index, -1], color="#2a8c62", linewidth=2,
                label=f"period {torque.shape[1]} settled")
        ax.set(xlabel="Joint angle [deg]", ylabel="Torque [N m]",
               title=f"{label}: {settled_offload[index]:.1f}% offload, "
                     f"{settled_rmse[index]:.1f} N m RMSE")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(output_dir, name, "fig05b_many_profile_examples.png"), dpi=200)
    plt.close(fig)


def encode_batch(dataset, spring_torque, scales, torque_scale):
    channels = np.stack((
        dataset["theta"] / scales[0], dataset["theta_dot"] / scales[1],
        dataset["theta_ddot"] / scales[2], dataset["target"] / torque_scale,
        spring_torque / torque_scale, (dataset["target"] - spring_torque) / torque_scale,
    ), axis=2)
    return channels.reshape(len(channels), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--profiles", type=int, default=100)
    parser.add_argument("--periods", type=int, default=10)
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--mechanics-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--constraint-min-order", type=float)
    parser.add_argument("--constraint-max-order", type=float)
    parser.add_argument("--output-name", default="period_adaptive_3d_60spring_bounded_extended_many_profiles")
    args = parser.parse_args()
    if args.profiles < 1 or args.periods < 2:
        parser.error("profiles must be positive and periods must be at least two")
    model, metadata = load_checkpoint(args.checkpoint)
    topology = load_spatial_topology(args.topology, torch.device(args.device))
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, args.relaxation_steps)
    profiles = generate_profile_parameters(np.random.default_rng(args.seed), args.profiles)
    dataset = build_period_dataset(
        profiles, angles, basis, float(metadata["period_seconds"]),
        int(metadata["samples_per_period"]), args.seed + 10_000,
        motion_mode=str(metadata["motion_mode"]),
        frequency_hz=1.0 / float(metadata["period_seconds"]),
        torque_scale=float(metadata["torque_scale"]),
    )
    stiffness = np.broadcast_to(
        np.asarray(metadata["initial_stiffness"], dtype=float)[None, :],
        (args.profiles, len(metadata["initial_stiffness"])),
    ).copy()
    hard_lower = hard_upper = None
    if args.constraint_min_order is not None or args.constraint_max_order is not None:
        if args.constraint_min_order is None or args.constraint_max_order is None:
            parser.error("both constraint order bounds must be supplied")
        initial = np.asarray(metadata["initial_stiffness"], dtype=float)
        hard_lower = initial * 10.0 ** args.constraint_min_order
        hard_upper = initial * 10.0 ** args.constraint_max_order
    torque_periods, stiffness_periods = [], []
    for period in range(args.periods):
        stiffness_periods.append(stiffness.copy())
        torque, _ = exact_period_torque(
            dataset, topology, stiffness[:, None, :], args.relaxation_steps,
            args.mechanics_batch_size, progress_interval=0,
        )
        measured = torque[:, 0]
        torque_periods.append(measured)
        observation = encode_batch(
            dataset, measured, np.asarray(metadata["motion_scales"]),
            float(metadata["torque_scale"]),
        )
        stiffness = controller_step(
            model, observation, float(metadata["min_k"]),
            metadata.get("stiffness_lower_bound"),
            metadata.get("stiffness_upper_bound"),
        )
        if hard_lower is not None:
            stiffness = np.clip(stiffness, hard_lower[None, :], hard_upper[None, :])
        print(f"Completed deployment period {period + 1}/{args.periods}", flush=True)
    torque = np.stack(torque_periods, axis=1)
    stiffness_history = np.stack(stiffness_periods, axis=1)
    target = dataset["target"][:, None, :]
    rmse = np.sqrt(np.mean((target - torque) ** 2, axis=2))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    baseline = trapezoid(
        np.abs(dataset["target"] * dataset["theta_dot"]), dataset["t"], axis=1
    )
    assisted = np.stack([
        trapezoid(np.abs((dataset["target"] - torque[:, p]) * dataset["theta_dot"]),
                  dataset["t"], axis=1)
        for p in range(args.periods)
    ], axis=1)
    offload = 100.0 * (1.0 - assisted / np.maximum(baseline[:, None], 1e-12))
    settled_rmse, settled_offload = rmse[:, -1], offload[:, -1]
    aggregate_offload = 100.0 * (
        1.0 - np.sum(assisted[:, -1]) / np.maximum(np.sum(baseline), 1e-12)
    )
    print(f"Profiles: {args.profiles} | periods/profile: {args.periods}")
    print(f"Default period mean RMSE: {np.mean(rmse[:, 0]):.3f} N*m")
    print(f"Settled period mean RMSE: {np.mean(settled_rmse):.3f} N*m")
    print(f"Settled RMSE p50/p90/p95: {np.percentile(settled_rmse, [50, 90, 95])}")
    print(f"Settled mean profile offload: {np.mean(settled_offload):.3f}%")
    print(f"Settled aggregate motor-work offload: {aggregate_offload:.3f}%")
    print(f"Settled offload p05/p50/p95: {np.percentile(settled_offload, [5, 50, 95])}")
    print(f"Negative-offload profiles: {100.0 * np.mean(settled_offload < 0):.2f}%")
    table = PROJECT_ROOT / "tables" / "period_adaptive_3d" / f"{args.output_name}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        fields = ["profile", "default_rmse_nm", "settled_rmse_nm", "settled_offload_pct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, profile in enumerate(profiles):
            writer.writerow({"profile": profile["name"], "default_rmse_nm": rmse[i, 0],
                             "settled_rmse_nm": settled_rmse[i],
                             "settled_offload_pct": settled_offload[i]})
    period_table = table.with_name(f"{args.output_name}_per_period.csv")
    with period_table.open("w", newline="", encoding="utf-8") as handle:
        fields = ["profile", "period", "policy", "rmse_nm", "offload_pct",
                  "baseline_motor_work_j", "assisted_motor_work_j"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, profile in enumerate(profiles):
            for period in range(args.periods):
                writer.writerow({
                    "profile": profile["name"], "period": period + 1,
                    "policy": "default" if period == 0 else "previous_period",
                    "rmse_nm": rmse[i, period], "offload_pct": offload[i, period],
                    "baseline_motor_work_j": baseline[i],
                    "assisted_motor_work_j": assisted[i, period],
                })
    summary_table = table.with_name(f"{args.output_name}_summary.csv")
    with summary_table.open("w", newline="", encoding="utf-8") as handle:
        fields = ["profiles", "periods", "default_mean_rmse_nm", "settled_mean_rmse_nm",
                  "settled_aggregate_offload_pct", "settled_mean_profile_offload_pct",
                  "negative_offload_profile_pct", "improved_rmse_profile_pct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "profiles": args.profiles, "periods": args.periods,
            "default_mean_rmse_nm": np.mean(rmse[:, 0]),
            "settled_mean_rmse_nm": np.mean(settled_rmse),
            "settled_aggregate_offload_pct": aggregate_offload,
            "settled_mean_profile_offload_pct": np.mean(settled_offload),
            "negative_offload_profile_pct": 100.0 * np.mean(settled_offload < 0),
            "improved_rmse_profile_pct": 100.0 * np.mean(settled_rmse < rmse[:, 0]),
        })
    np.savez_compressed(
        table.with_name(f"{args.output_name}_complete_data.npz"),
        time=dataset["t"], theta=dataset["theta"], theta_dot=dataset["theta_dot"],
        theta_ddot=dataset["theta_ddot"], target_torque=dataset["target"],
        spring_torque=torque, residual_motor_torque=target - torque,
        stiffness=stiffness_history, rmse=rmse, offload_pct=offload,
        baseline_motor_work_j=baseline, assisted_motor_work_j=assisted,
    )
    save_benchmark_figures(
        PROJECT_ROOT / "plots" / "period_adaptive_3d", args.output_name,
        dataset, torque, rmse, offload,
    )
    print(f"Saved {table}")
    print(f"Saved {period_table}, {summary_table}, and complete compressed arrays")


if __name__ == "__main__":
    main()
