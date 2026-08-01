"""Generate exact-mechanics figures for a profile-conditioned passive 3D model."""

from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES
from benchmark_profile_passive_3d import relaxed_spatial_profile_torque, spatial_initial_basis
from mechanics_3d import load_spatial_topology
from profile_generator import PROFILE_FAMILIES, generate_classified_profile_parameters
from train_profile_conditioned_passive import (
    build_profile_dataset,
    predict_profile_stiffness,
    summary_rows,
)


def load_checkpoint(path):
    saved = np.load(path, allow_pickle=False)
    model = {name: saved[name] for name in ("w1", "b1", "w2", "b2")}
    return saved, model


def representative_indices(profiles, rows):
    """Choose low/median offload examples from each roughness family."""
    selected = []
    for family in PROFILE_FAMILIES:
        indices = [i for i, profile in enumerate(profiles) if profile["family"] == family]
        ordered = sorted(indices, key=lambda i: rows[i]["offload_pct"])
        selected.extend((ordered[len(ordered) // 4], ordered[len(ordered) // 2]))
    return selected


def plot_torque_angle(path, profiles, dataset, torque, rows, indices):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, index in zip(axes.flat, indices):
        angle = np.rad2deg(dataset["theta"][index])
        target = dataset["target"][index]
        spring = torque[index]
        motor = target - spring
        order = np.argsort(angle)
        axis.plot(angle[order], target[order], "k--", linewidth=2.2, label="target torque")
        axis.plot(angle[order], spring[order], color="tab:blue", linewidth=2.0, label="passive spring")
        axis.plot(angle[order], motor[order], color="tab:red", linewidth=1.7, label="residual motor")
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.set_title(
            f"{profiles[index]['family']} · offload {rows[index]['offload_pct']:.1f}%\n"
            f"RMSE {rows[index]['residual_rmse_nm']:.1f} Nm"
        )
        axis.set_xlabel("joint angle [deg]")
        axis.set_ylabel("torque [Nm]")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_time_traces(path, profiles, dataset, torque, rows, indices):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, index in zip(axes.flat, indices):
        time = dataset["t"][index]
        target = dataset["target"][index]
        spring = torque[index]
        motor = target - spring
        axis.plot(time, target, "k--", linewidth=2.1, label="target torque")
        axis.plot(time, spring, color="tab:blue", linewidth=1.9, label="passive spring")
        axis.plot(time, motor, color="tab:red", linewidth=1.5, label="residual motor")
        axis.set_title(
            f"{profiles[index]['family']} · offload {rows[index]['offload_pct']:.1f}%"
        )
        axis.set_xlabel("time [s]")
        axis.set_ylabel("torque [Nm]")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_stiffness(path, profiles, stiffness, rows, indices):
    values = stiffness[indices]
    figure, axis = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xlabel("spring index")
    axis.set_ylabel("held-out profile")
    axis.set_yticks(
        np.arange(len(indices)),
        [
            f"{profiles[i]['family']} ({rows[i]['offload_pct']:.1f}%)"
            for i in indices
        ],
    )
    axis.set_title("One fixed stiffness vector per torque–angle profile")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("stiffness [N/m]")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profiles-per-family", type=int, default=2000)
    parser.add_argument("--test-profiles-per-family", type=int, default=400)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--mechanics-batch-size", type=int, default=1024)
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()
    saved, model = load_checkpoint(args.checkpoint)
    topology_path = Path(str(saved["topology"]))
    if not topology_path.is_absolute():
        topology_path = Path.cwd() / topology_path
    topology = load_spatial_topology(topology_path, args.device)
    topology["nonlinear_power"] = (
        int(saved["nonlinear_power"]) if "nonlinear_power" in saved else 1
    )
    topology["nonlinear_ratio"] = (
        float(saved["nonlinear_ratio"]) if "nonlinear_ratio" in saved else 0.0
    )
    topology["nonlinear_reference_extension"] = (
        float(saved["nonlinear_reference_extension"])
        if "nonlinear_reference_extension" in saved else 0.6
    )
    relaxation_steps = int(saved["relaxation_steps"])
    seed = int(saved["seed"])
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, relaxation_steps)
    rng = np.random.default_rng(seed)
    generate_classified_profile_parameters(rng, args.profiles_per_family)
    profiles = generate_classified_profile_parameters(rng, args.test_profiles_per_family)
    dataset = build_profile_dataset(
        profiles, angles, basis, args.duration, args.samples, seed + 30_000
    )
    stiffness = predict_profile_stiffness(
        model, dataset, float(saved["min_k"]), 1.0, unbounded_stiffness=True
    )
    torque, _ = relaxed_spatial_profile_torque(
        dataset, topology, stiffness, relaxation_steps, args.mechanics_batch_size
    )
    rows = summary_rows(profiles, dataset, torque, stiffness)
    indices = representative_indices(profiles, rows)
    stem = args.output_stem or args.checkpoint.stem
    output = PROJECT_ROOT / "plots" / "profile_conditioned_passive_3d"
    plot_torque_angle(output / f"{stem}_torque_angle.png", profiles, dataset, torque, rows, indices)
    plot_time_traces(output / f"{stem}_time_traces.png", profiles, dataset, torque, rows, indices)
    plot_stiffness(output / f"{stem}_stiffness_heatmap.png", profiles, stiffness, rows, indices)
    print(f"Saved {output / f'{stem}_torque_angle.png'}")
    print(f"Saved {output / f'{stem}_time_traces.png'}")
    print(f"Saved {output / f'{stem}_stiffness_heatmap.png'}")


if __name__ == "__main__":
    main()
