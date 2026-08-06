"""Audit a stiffness checkpoint across relaxed-mechanics convergence depths."""

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

from adaptive_model import ANGLE_DEGREES, forward, spring_torque_basis
from profile_generator import (
    PROFILE_FAMILIES,
    generate_classified_profile_parameters,
)
from topology_loader import load_network
import train_adaptive_dataset as trainer
from train_adaptive_dataset import (
    build_dataset,
    generate_periodicity_profiles,
    summarize_profiles,
    torch_prescribed_positions,
    torch_relax_positions,
    torch_topology_data,
)


DEFAULT_TOPOLOGY = (
    PROJECT_ROOT
    / "topologies"
    / "adaptive_stiffness"
    / "internal_fan_20_spring_model.json"
)


def scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    return value.item() if value.shape == () else value


def load_checkpoint(path):
    with np.load(path, allow_pickle=True) as data:
        model = {key: np.asarray(data[key], dtype=float) for key in ("w1", "b1", "w2", "b2")}
        metadata = {key: scalar(data, key) for key in data.files if key not in model}
    return model, metadata


def reconstruct_test_profiles(metadata, limit_per_family):
    seed = int(metadata.get("seed", 11))
    # Older checkpoints did not store seed; all active stiffness runs used 11.
    rng = np.random.default_rng(seed)
    train_count = int(metadata["profiles_per_family"])
    test_count = int(metadata["test_profiles_per_family"])
    classification = str(metadata.get("classification_mode", "roughness"))
    duration = float(metadata["duration"])
    samples = int(metadata["samples"])
    if classification.startswith("periodicity-"):
        periodicity_class = classification.removeprefix("periodicity-")
        generate_periodicity_profiles(
            rng, train_count, duration, samples, seed + 1_000, periodicity_class
        )
        profiles = generate_periodicity_profiles(
            rng, test_count, duration, samples, seed + 2_000, periodicity_class
        )
    else:
        generate_classified_profile_parameters(rng, train_count)
        profiles = generate_classified_profile_parameters(rng, test_count)
    if limit_per_family <= 0:
        return profiles
    selected = []
    counts = {}
    for profile in profiles:
        family = profile["family"]
        if counts.get(family, 0) < limit_per_family:
            selected.append(profile)
            counts[family] = counts.get(family, 0) + 1
    return selected


def relaxed_torque_and_force_residual(
    topology, theta, stiffness, relaxation_steps, cubic_ratio, cubic_reference
):
    positions = torch_prescribed_positions(topology, theta)
    positions = torch_relax_positions(
        topology, positions, stiffness, relaxation_steps, 0.03
    )
    a, b = topology["spring_a"], topology["spring_b"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    direction = delta / length.unsqueeze(2)
    stretch = length - topology["rest_lengths"].unsqueeze(0)
    cubic = cubic_ratio / max(cubic_reference**2, 1e-12)
    force_on_a = (
        stiffness * stretch + stiffness * cubic * stretch**3
    ).unsqueeze(2) * direction

    node_force = torch.zeros_like(positions)
    node_force.index_add_(1, a, force_on_a)
    node_force.index_add_(1, b, -force_on_a)
    internal = topology["internal_indices"]
    if internal.numel():
        force_residual = torch.max(
            torch.linalg.norm(node_force[:, internal, :], dim=2), dim=1
        ).values
    else:
        force_residual = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)

    torque = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)
    limb2 = set(int(index) for index in topology["limb2_indices"].detach().cpu().numpy())
    for spring_index in range(len(a)):
        node_a, node_b = int(a[spring_index]), int(b[spring_index])
        if node_a in limb2:
            r = positions[:, node_a, :]
            f = force_on_a[:, spring_index, :]
            torque += r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]
        if node_b in limb2:
            r = positions[:, node_b, :]
            f = -force_on_a[:, spring_index, :]
            torque += r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]
    return torque, force_residual


def audit_rollout(
    model, dataset, topology_path, min_k, max_k, depth, device,
    cubic_ratio, cubic_reference,
):
    profiles = len(dataset["target"]) // dataset["samples_per_profile"]
    samples = dataset["samples_per_profile"]
    window = dataset["window_size"]
    motion = dataset["features"].reshape(profiles, samples, -1)
    target = dataset["target"].reshape(profiles, samples)
    theta = dataset["theta"].reshape(profiles, samples)
    update_mask = dataset["update_mask"].reshape(profiles, samples)
    history = np.zeros((profiles, window, 3), dtype=float)
    predicted = np.empty((profiles, samples), dtype=float)
    residuals = np.empty((profiles, samples), dtype=float)
    held = None

    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, device)
    for sample_index in range(samples):
        inputs = np.hstack((motion[:, sample_index, :], history.reshape(profiles, -1)))
        candidate, _ = forward(model, inputs, min_k, max_k)
        stiffness = candidate if held is None else np.where(
            update_mask[:, sample_index, None], candidate, held
        )
        held = stiffness
        torque, force_residual = relaxed_torque_and_force_residual(
            topology,
            torch.as_tensor(theta[:, sample_index], dtype=torch.float32, device=device),
            torch.as_tensor(stiffness, dtype=torch.float32, device=device),
            depth,
            cubic_ratio,
            cubic_reference,
        )
        torque_np = torque.detach().cpu().numpy()
        predicted[:, sample_index] = torque_np
        residuals[:, sample_index] = force_residual.detach().cpu().numpy()
        motor = target[:, sample_index] - torque_np
        realized = np.stack((target[:, sample_index], torque_np, motor), axis=1)
        realized /= max(dataset["torque_scale"], 1e-9)
        history = np.concatenate((history[:, 1:, :], realized[:, None, :]), axis=1)
    return predicted.reshape(-1), residuals.reshape(-1)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--depths", type=int, nargs="+", default=[30, 80, 160, 300])
    parser.add_argument(
        "--profiles-per-family",
        type=int,
        default=10,
        help="Audit subset per held-out family; use 0 for the complete held-out set.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--cubic-ratio", type=float, default=None)
    parser.add_argument("--cubic-reference-extension-mm", type=float, default=None)
    parser.add_argument("--rmse-tolerance-nm", type=float, default=0.25)
    parser.add_argument("--offload-tolerance-pct", type=float, default=0.25)
    parser.add_argument("--force-tolerance-n", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tables" / "mechanics_audits")
    args = parser.parse_args()
    if any(depth < 0 for depth in args.depths):
        parser.error("Relaxation depths must be nonnegative")
    depths = sorted(set(args.depths))
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")

    model, metadata = load_checkpoint(args.checkpoint)
    cubic_ratio = (
        float(args.cubic_ratio)
        if args.cubic_ratio is not None
        else float(metadata.get("cubic_ratio", 0.0))
    )
    cubic_reference_mm = (
        float(args.cubic_reference_extension_mm)
        if args.cubic_reference_extension_mm is not None
        else float(metadata.get("cubic_reference_extension_mm", 50.0))
    )
    if cubic_reference_mm <= 0.0:
        parser.error("Cubic reference extension must be positive")
    if cubic_ratio and "cubic_ratio" not in metadata and args.cubic_ratio is None:
        parser.error("Older cubic checkpoints require an explicit --cubic-ratio")
    trainer.EXPERIMENT_CUBIC_RATIO = cubic_ratio
    trainer.EXPERIMENT_CUBIC_REFERENCE_EXTENSION = cubic_reference_mm / 1000.0

    profiles = reconstruct_test_profiles(metadata, args.profiles_per_family)
    network, _ = load_network(args.topology)
    angles = np.radians(ANGLE_DEGREES)
    basis = spring_torque_basis(network, angles, relax_internal=True)
    scales = {
        "theta": float(metadata["theta_scale"]),
        "theta_dot": float(metadata["theta_dot_scale"]),
        "theta_ddot": float(metadata["theta_ddot_scale"]),
        "torque": float(metadata["torque_scale"]),
    }
    dataset = build_dataset(
        profiles,
        angles,
        basis,
        float(metadata["duration"]),
        int(metadata["samples"]),
        int(metadata["window_size"]),
        scales,
        int(metadata.get("seed", 11)) + 30_000,
        stiffness_update_mode=str(metadata.get("stiffness_update_mode", "timestep")),
        include_profile_descriptor=bool(metadata.get("include_profile_descriptor", False)),
        motion_mode=str(metadata.get("motion_mode", "randomized")),
        fixed_frequency_hz=metadata.get("fixed_frequency_hz"),
    )

    results = []
    predictions = {}
    deepest = depths[-1]
    for depth in depths:
        print(f"Auditing relaxation depth {depth}...")
        predicted, force_residual = audit_rollout(
            model,
            dataset,
            args.topology,
            float(metadata["min_k"]),
            float(metadata["max_k"]),
            depth,
            device,
            cubic_ratio,
            cubic_reference_mm / 1000.0,
        )
        predictions[depth] = predicted
        rows = summarize_profiles(
            profiles,
            dataset,
            predicted,
            float(metadata.get("motoring_efficiency", 1.0)),
            float(metadata.get("regen_efficiency", 0.0)),
        )
        results.append({
            "relaxation_steps": depth,
            "profiles": len(profiles),
            "mean_rmse_nm": float(np.mean([row["rmse_nm"] for row in rows])),
            "mean_offload_pct": float(np.mean([row["offload_pct"] for row in rows])),
            "mean_force_residual_n": float(np.mean(force_residual)),
            "max_force_residual_n": float(np.max(force_residual)),
            "torque_rmse_vs_deepest_nm": 0.0,
            "max_torque_difference_vs_deepest_nm": 0.0,
        })

    reference = predictions[deepest]
    for result in results:
        delta = predictions[result["relaxation_steps"]] - reference
        result["torque_rmse_vs_deepest_nm"] = float(np.sqrt(np.mean(delta**2)))
        result["max_torque_difference_vs_deepest_nm"] = float(np.max(np.abs(delta)))
    for current, following in zip(results[:-1], results[1:]):
        current["rmse_change_to_next_nm"] = abs(
            following["mean_rmse_nm"] - current["mean_rmse_nm"]
        )
        current["offload_change_to_next_pct"] = abs(
            following["mean_offload_pct"] - current["mean_offload_pct"]
        )
    results[-1]["rmse_change_to_next_nm"] = 0.0
    results[-1]["offload_change_to_next_pct"] = 0.0
    for result in results:
        result["passes_force_tolerance"] = (
            result["max_force_residual_n"] <= args.force_tolerance_n
        )
        result["passes_next_depth_stability"] = (
            result["rmse_change_to_next_nm"] <= args.rmse_tolerance_nm
            and result["offload_change_to_next_pct"] <= args.offload_tolerance_pct
        )

    stem = args.checkpoint.stem
    csv_path = args.output_dir / f"{stem}_mechanics_audit.csv"
    plot_path = args.output_dir / f"{stem}_mechanics_audit.png"
    write_rows(csv_path, results)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    x = [row["relaxation_steps"] for row in results]
    axes[0, 0].plot(x, [row["mean_rmse_nm"] for row in results], marker="o")
    axes[0, 0].set_ylabel("Mean RMSE [Nm]")
    axes[0, 1].plot(x, [row["mean_offload_pct"] for row in results], marker="o")
    axes[0, 1].set_ylabel("Mean offload [%]")
    axes[1, 0].semilogy(x, [max(row["max_force_residual_n"], 1e-12) for row in results], marker="o")
    axes[1, 0].axhline(args.force_tolerance_n, color="tab:red", linestyle="--", label="tolerance")
    axes[1, 0].set_ylabel("Max internal force residual [N]")
    axes[1, 0].legend()
    axes[1, 1].semilogy(x, [max(row["torque_rmse_vs_deepest_nm"], 1e-12) for row in results], marker="o")
    axes[1, 1].set_ylabel(f"Torque RMSE vs {deepest} steps [Nm]")
    for axis in axes.flat:
        axis.set_xlabel("Relaxation steps")
        axis.grid(True, alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print()
    print("steps | RMSE Nm | offload % | max force N | torque RMSE vs deepest")
    for row in results:
        print(
            f"{row['relaxation_steps']:5d} | {row['mean_rmse_nm']:7.3f} | "
            f"{row['mean_offload_pct']:9.3f} | {row['max_force_residual_n']:11.4g} | "
            f"{row['torque_rmse_vs_deepest_nm']:22.4f}"
        )
    print(f"Saved audit table: {csv_path}")
    print(f"Saved audit plot:  {plot_path}")


if __name__ == "__main__":
    main()
