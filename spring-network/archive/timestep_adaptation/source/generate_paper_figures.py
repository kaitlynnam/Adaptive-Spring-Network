"""Generate compact paper figures from the currently preserved result tables."""

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from profile_generator import generate_classified_profile_parameters
from train_adaptive_dataset import generate_motion_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "plots" / "paper_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def box(axis, xy, width, height, text, color):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.025",
        facecolor=color, edgecolor="#263238", linewidth=1.5,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=10,
    )


def pipeline():
    figure, axis = plt.subplots(figsize=(12, 4.1))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    items = [
        (0.3, "Joint motion\nθ, θ̇, θ̈", "#dbeafe"),
        (
            2.6,
            "Causal history\npast target, spring,\nand residual motor torque",
            "#e0f2fe",
        ),
        (4.9, "Neural network\nMLP controller", "#fef3c7"),
        (7.2, "48/56 stiffness\ncommands", "#fde68a"),
        (9.5, "3D spring network\n+ node relaxation", "#dcfce7"),
    ]
    for x, label, color in items:
        box(axis, (x, 1.55), 1.8, 1.05, label, color)
    for left, right in zip(items[:-1], items[1:]):
        axis.annotate(
            "", xy=(right[0] - 0.08, 2.08), xytext=(left[0] + 1.88, 2.08),
            arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "#374151"},
        )
    axis.annotate(
        "spring torque + residual motor torque",
        xy=(3.5, 1.42), xytext=(10.4, 0.72),
        ha="center", va="center",
        arrowprops={
            "arrowstyle": "->", "linewidth": 1.8,
            "connectionstyle": "arc3,rad=-0.22", "color": "#7c3aed",
        },
        color="#5b21b6", fontsize=10,
    )
    axis.set_title("Causal Adaptive-Stiffness Control Pipeline", fontsize=15, pad=8)
    figure.savefig(OUTPUT / "fig02_neural_to_spring_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def torque_profiles():
    # Recreate the held-out profile pool used by the standard full training
    # command (seed 11, 2000 training profiles and 400 test profiles per
    # historical reporting group). Group labels are intentionally not shown.
    rng = np.random.default_rng(11)
    generate_classified_profile_parameters(rng, 2000)
    held_out = generate_classified_profile_parameters(rng, 400)
    # Use ordinary, well-spaced examples from that pool so the figure is
    # representative without near-vertical segments caused by clustered knots.
    profiles = [held_out[index] for index in (30, 44, 52)]
    angle_figure, angle_axes = plt.subplots(
        1, 3, figsize=(14, 4.2), constrained_layout=True,
    )
    time_figure, time_axes = plt.subplots(
        1, 3, figsize=(14, 4.2), constrained_layout=True,
    )
    for index, params in enumerate(profiles):
        tau_knots = params["knots_tau"]
        dense_angle = np.linspace(-45.0, 45.0, 500)
        dense_torque = np.interp(
            np.deg2rad(dense_angle), params["knots_theta"], tau_knots,
        )
        time, _, _, _, target_time = generate_motion_trajectory(
            params,
            duration=20.0,
            samples=640,
            seed=30_011 + index,
            motion_mode="triangular",
            fixed_frequency_hz=0.25,
        )
        angle_axes[index].plot(
            dense_angle,
            dense_torque,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="target",
        )
        time_axes[index].plot(
            time,
            target_time,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="target",
            zorder=4,
        )
        angle_axes[index].set_title(f"Profile {chr(65 + index)}")
        time_axes[index].set_title(f"Profile {chr(65 + index)}")
        angle_axes[index].set_xlabel("joint angle [deg]")
        time_axes[index].set_xlabel("time [s]")
        angle_axes[index].grid(alpha=0.25)
        time_axes[index].grid(alpha=0.25)
        angle_axes[index].set_xlim(-45.0, 45.0)
        time_axes[index].set_xlim(0.0, 20.0)
        for axis in (angle_axes[index], time_axes[index]):
            axis.axhline(0.0, color="0.7", linewidth=1.0)
    angle_axes[0].set_ylabel("target torque [N·m]")
    time_axes[0].set_ylabel("target torque [N·m]")
    angle_figure.suptitle(
        "Held-Out Piecewise-Linear Torque–Angle Profiles", fontsize=15,
    )
    time_figure.suptitle(
        "Corresponding Held-Out Torque–Time Profiles at 0.25 Hz", fontsize=15,
    )
    angle_figure.savefig(OUTPUT / "fig03a_torque_angle_profiles.png", dpi=220)
    time_figure.savefig(OUTPUT / "fig03b_torque_time_profiles.png", dpi=220)
    plt.close(angle_figure)
    plt.close(time_figure)


def primary_comparison():
    path = (
        PROJECT_ROOT / "tables" / "spatial"
        / "global_56s_c0131_screen300_mechanics_comparison.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = ["Fixed stiffness\nbaseline", "Adaptive\nstiffness"]
    offload = [float(row["mean_offload_pct"]) for row in rows]
    rmse = [float(row["mean_rmse_nm"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), constrained_layout=True)
    bars = axes[0].bar(labels, offload, color=["#94a3b8", "#2563eb"])
    axes[1].bar(labels, rmse, color=["#94a3b8", "#2563eb"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("mean motor-energy offload [%]")
    axes[1].set_ylabel("mean torque RMSE [N·m]")
    axes[0].set_title("Motor offload")
    axes[1].set_title("Torque tracking error")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, offload):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value + (1 if value >= 0 else -3),
            f"{value:.1f}%", ha="center",
        )
    figure.suptitle(
        "Preliminary 56-Spring 3D Performance (30 Held-Out Profiles)",
        fontsize=14,
    )
    figure.savefig(OUTPUT / "fig05_primary_performance_comparison_preliminary.png", dpi=220)
    plt.close(figure)


def adaptive_behavior():
    import importlib.util
    renderer_path = (
        PROJECT_ROOT / "01_core_model" / "render_spatial_topology_interactive.py"
    )
    spec = importlib.util.spec_from_file_location("spatial_viewer", renderer_path)
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    topology = viewer.load_spatial_topology(viewer.DEFAULT_TOPOLOGY, "cuda")
    dataset, spring_torque, stiffness, _, _ = viewer.learned_demo_rollout(
        topology,
        viewer.DEFAULT_CHECKPOINT,
        profile_index=28,
        use_heldout_profile=True,
    )
    time = dataset["t"]
    target = dataset["target"]
    residual = target - spring_torque
    figure, axes = plt.subplots(
        3, 1, figsize=(11, 8.2),
        gridspec_kw={"height_ratios": [0.8, 1.2, 1.35]},
        constrained_layout=True,
    )
    angle_deg = np.degrees(dataset["theta"])
    angle_order = np.argsort(angle_deg)
    axes[0].plot(
        angle_deg[angle_order],
        target[angle_order],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="target torque",
        zorder=4,
    )
    axes[0].scatter(
        angle_deg,
        spring_torque,
        s=18,
        color="#2563eb",
        alpha=0.75,
        label="spring torque",
        zorder=3,
    )
    axes[0].set(xlabel="joint angle [deg]", ylabel="torque [N·m]")
    axes[0].legend(frameon=False)
    axes[1].plot(time, target, "k--", linewidth=2, label="target torque")
    axes[1].plot(
        time, spring_torque, color="#2563eb", linewidth=1.8,
        label="spring torque",
    )
    axes[1].plot(
        time, residual, color="#f97316", linewidth=1.5,
        label="residual motor torque",
    )
    axes[1].set(xlabel="time [s]", ylabel="torque [N·m]")
    axes[1].legend(ncol=3, frameon=False)

    # Place spatially neighboring springs in neighboring heatmap rows. The
    # ordering is based on neutral-position spring midpoints and does not alter
    # controller outputs or mechanics.
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    neutral = topology["local_positions"].detach().cpu().numpy()
    spring_a = topology["spring_a"].detach().cpu().numpy()
    spring_b = topology["spring_b"].detach().cpu().numpy()
    midpoints = 0.5 * (neutral[spring_a] + neutral[spring_b])
    midpoint_distances = pdist(midpoints)
    spatial_order = leaves_list(
        optimal_leaf_ordering(
            linkage(midpoint_distances, method="average"),
            midpoint_distances,
        )
    )
    image = axes[2].imshow(
        stiffness[:, spatial_order].T, origin="lower", aspect="auto",
        extent=[time[0], time[-1], 1, stiffness.shape[1]],
        cmap="turbo", vmin=1, vmax=800,
    )
    axes[2].set(
        xlabel="time [s]",
        ylabel="springs (spatially ordered)",
    )
    colorbar = figure.colorbar(image, ax=axes[2], pad=0.015)
    colorbar.set_label("commanded stiffness [N/m]")
    for axis in axes[:2]:
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Example Causal Adaptive Behavior — One Held-Out 48-Spring Test",
        fontsize=14,
    )
    figure.savefig(OUTPUT / "fig04_example_adaptive_behavior.png", dpi=220)
    plt.close(figure)


def convergence():
    figure, axis = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    for law, color in (("linear", "#2563eb"), ("cubic", "#dc2626")):
        path = (
            PROJECT_ROOT / "tables" / "mechanics_audits"
            / f"stiffness_{law}_refreshed_relax300_mechanics_audit.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        steps = [int(row["relaxation_steps"]) for row in rows]
        offload = [float(row["mean_offload_pct"]) for row in rows]
        axis.plot(
            steps, offload, "o-", linewidth=2, markersize=6,
            color=color, label=f"{law} springs",
        )
    axis.set(
        xlabel="internal-node relaxation steps",
        ylabel="mean motor-energy offload [%]",
        title="Mechanical Convergence of Reported Offload",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(OUTPUT / "fig06_offload_relaxation_convergence.png", dpi=220)
    plt.close(figure)


def seed_robustness():
    seeds = (401, 503, 607)
    paths = [
        PROJECT_ROOT / "tables" / "spatial"
        / f"paper_global56_linear_seed{seed}_mechanics_comparison.csv"
        for seed in seeds
    ]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Independent paper runs are incomplete: " + ", ".join(missing)
        )

    offload, rmse = [], []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        adaptive = next(row for row in rows if row["model"] == "adaptive_spatial")
        offload.append(float(adaptive["mean_offload_pct"]))
        rmse.append(float(adaptive["mean_rmse_nm"]))

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), constrained_layout=True)
    axes[0].scatter(seeds, offload, s=65, color="#2563eb", zorder=3)
    axes[1].scatter(seeds, rmse, s=65, color="#f97316", zorder=3)
    axes[0].axhline(np.mean(offload), color="#1e3a8a", linestyle="--")
    axes[1].axhline(np.mean(rmse), color="#9a3412", linestyle="--")
    axes[0].set(ylabel="mean offload [%]", title="Motor offload")
    axes[1].set(ylabel="mean torque RMSE [N·m]", title="Tracking error")
    for axis in axes:
        axis.set_xlabel("independent training seed")
        axis.set_xticks(seeds)
        axis.grid(alpha=0.25)
    figure.suptitle("Multiple-Seed Robustness of the Selected 56-Spring Topology")
    figure.savefig(OUTPUT / "fig06_multiple_seed_robustness.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    pipeline()
    torque_profiles()
    adaptive_behavior()
    primary_comparison()
    convergence()
    if all(
        (
            PROJECT_ROOT / "tables" / "spatial"
            / f"paper_global56_linear_seed{seed}_mechanics_comparison.csv"
        ).exists()
        for seed in (401, 503, 607)
    ):
        seed_robustness()
    print(OUTPUT)
