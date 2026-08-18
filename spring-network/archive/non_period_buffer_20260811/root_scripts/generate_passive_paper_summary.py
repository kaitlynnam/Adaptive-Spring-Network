"""Generate canonical 60-spring paper summary figures from completed seeds."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent
PROJECT = REPO / "spring-network"
TABLES = PROJECT / "tables" / "profile_conditioned_passive_3d"
OUTPUT = PROJECT / "plots" / "profile_conditioned_passive_3d"
TOPOLOGY = PROJECT / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"
SEEDS = (101,)
OUTPUT_STEM = "passive_skin60"


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pipeline_figure():
    figure, axis = plt.subplots(figsize=(13, 5.0), constrained_layout=True)
    axis.axis("off")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    labels = [
        "Five-knot\ntorque–angle profile",
        "Profile-conditioned\nMLP",
        "60 positive fixed\nstiffnesses",
        "Relaxed 3D\nspring mechanics",
        "Passive torque +\nresidual motor torque",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(xs, labels):
        axis.text(
            x, 0.68, label, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f1f8", edgecolor="#315a76"),
        )
    for left, right in zip(xs[:-1], xs[1:]):
        axis.annotate("", xy=(right - 0.075, 0.68), xytext=(left + 0.075, 0.68),
                      arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#303030"))
    loss_x, loss_y = 0.71, 0.25
    axis.text(
        loss_x, loss_y,
        "Training loss\nresidual torque MSE + motor-work offload term",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.65", facecolor="#fde8dc", edgecolor="#b4532a"),
    )
    axis.annotate(
        "compare with target",
        xy=(loss_x + 0.08, loss_y + 0.09), xytext=(xs[-1], 0.58),
        ha="center", fontsize=9.5,
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#b4532a"),
    )
    axis.annotate(
        "",
        xy=(xs[1], 0.59), xytext=(loss_x - 0.14, loss_y + 0.02),
        arrowprops=dict(
            arrowstyle="->", linewidth=2.0, color="#b4532a",
            connectionstyle="arc3,rad=-0.25",
        ),
    )
    axis.text(
        0.61, 0.39,
        "Backpropagation\nupdates MLP weights",
        ha="center", va="center", fontsize=9.5, color="#8a3f20",
        bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.95),
    )
    axis.set_title("Profile-Conditioned Passive 60-Spring Pipeline", fontsize=15, pad=12)
    figure.savefig(OUTPUT / "fig02_passive_control_pipeline.png", dpi=220)
    plt.close(figure)


def convergence_figure():
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for seed in SEEDS:
        rows = read_rows(TABLES / f"{OUTPUT_STEM}_seed{seed}_training_history.csv")
        iteration = np.asarray([float(row["iteration"]) for row in rows])
        rmse = np.asarray([float(row["residual_rmse_nm"]) for row in rows])
        offload = np.asarray([float(row["offload_pct"]) for row in rows])
        axes[0].plot(iteration, rmse, label=f"seed {seed}", linewidth=1.5)
        axes[1].plot(iteration, offload, label=f"seed {seed}", linewidth=1.5)
    axes[0].set(title="Residual torque error", xlabel="training iteration", ylabel="RMSE [Nm]")
    axes[1].set(title="Estimated motor-work offload", xlabel="training iteration", ylabel="offload [%]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(OUTPUT / "fig06_training_and_correction_convergence.png", dpi=220)
    plt.close(figure)


def performance_figure():
    rows = read_rows(TABLES / f"{OUTPUT_STEM}_seed101_test_results.csv")
    offload = np.asarray([float(row["offload_pct"]) for row in rows])
    rmse = np.asarray([float(row["residual_rmse_nm"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(offload, bins=30, color="#3b82a0")
    axes[1].hist(rmse, bins=30, color="#ca6f4b")
    axes[0].set(title="Held-out motor-work offload", xlabel="offload [%]", ylabel="profiles")
    axes[1].set(title="Held-out residual torque error", xlabel="RMSE [Nm]", ylabel="profiles")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(OUTPUT / "fig07_heldout_offload_and_residual_error.png", dpi=220)
    plt.close(figure)



def topology_figure():
    sys.path.insert(0, str(PROJECT / "01_core_model"))
    from render_spatial_topology import render
    render(TOPOLOGY, OUTPUT / "fig01_60spring_topology.png", angle_degrees=25.0)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    required = [TABLES / f"{OUTPUT_STEM}_seed{seed}_test_results.csv" for seed in SEEDS]
    required += [TABLES / f"{OUTPUT_STEM}_seed{seed}_training_history.csv" for seed in SEEDS]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Complete the canonical seed first: " + ", ".join(map(str, missing)))
    topology_figure()
    pipeline_figure()
    convergence_figure()
    performance_figure()


if __name__ == "__main__":
    main()
