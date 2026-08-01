"""Train a profile-to-one-passive-stiffness-vector controller in 3D mechanics."""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES, save_model
from benchmark_profile_passive_3d import (
    relaxed_spatial_profile_torque,
    spatial_initial_basis,
)
from mechanics_3d import load_spatial_topology, torque_components
from profile_generator import (
    PROFILE_CLASSIFICATION,
    PROFILE_FAMILIES,
    generate_classified_profile_parameters,
)
from train_profile_conditioned_passive import (
    build_profile_dataset,
    expand_profile_stiffness,
    predict_profile_stiffness,
    summary_rows,
    train_profile_model,
    write_rows,
    write_stiffness_rows,
)

DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial" / "internal_fan_3d_60_spring.json"
)


def subset_profile_dataset(dataset, indices):
    indices = np.asarray(indices, dtype=int)
    result = dict(dataset)
    profile_count = len(dataset["profile_features"])
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and value.shape[0] == profile_count:
            result[key] = value[indices]
    return result


def correction_indices(profile_count, sample_count, requested_profiles, requested_samples, rng):
    """Select the complete dataset by default; positive limits enable debug subsets."""
    if requested_profiles < 0 or requested_samples < 0:
        raise ValueError("mechanics correction limits cannot be negative")
    if requested_profiles == 0 or requested_profiles >= profile_count:
        profile_indices = np.arange(profile_count)
    else:
        profile_indices = rng.choice(
            profile_count, size=requested_profiles, replace=False
        )
    if requested_samples == 0 or requested_samples >= sample_count:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.sort(
            rng.choice(sample_count, size=requested_samples, replace=False)
        )
    return profile_indices, sample_indices


def refresh_mlp_spatial_basis(
    model,
    dataset,
    topology,
    min_stiffness,
    relaxation_steps,
    batch_size=512,
):
    """Re-relax 3D mechanics at the MLP's one-vector-per-profile predictions."""
    stiffness = predict_profile_stiffness(
        model, dataset, min_stiffness, 1.0, unbounded_stiffness=True
    )
    schedule = expand_profile_stiffness(
        stiffness, dataset["samples_per_profile"]
    ).reshape(-1, stiffness.shape[1])
    theta = dataset["theta"].reshape(-1)
    components = np.empty_like(schedule)
    device = topology["local_positions"].device
    for start in range(0, len(theta), batch_size):
        stop = min(start + batch_size, len(theta))
        theta_batch = torch.as_tensor(
            theta[start:stop], dtype=torch.float32, device=device
        )
        stiffness_batch = torch.as_tensor(
            schedule[start:stop].copy(), dtype=torch.float32, device=device
        )
        values = torque_components(
            topology, theta_batch, stiffness_batch, relaxation_steps
        )
        components[start:stop] = values.detach().cpu().numpy()
    refreshed = dict(dataset)
    refreshed["basis"] = (
        components / np.maximum(schedule, 1e-6)
    ).reshape(dataset["basis"].shape)
    return refreshed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--profiles-per-family", type=int, default=2000)
    parser.add_argument("--test-profiles-per-family", type=int, default=400)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--energy-weight", type=float, default=30.0)
    parser.add_argument("--min-stiffness", type=float, default=0.0)
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument(
        "--nonlinear-power",
        type=int,
        choices=[1, 3, 4, 5],
        default=1,
        help="Restoring force power; 1 selects linear springs.",
    )
    parser.add_argument(
        "--nonlinear-ratio",
        type=float,
        default=0.0,
        help="Nonnegative power-law strength at the reference extension.",
    )
    parser.add_argument("--nonlinear-reference-extension", type=float, default=0.6)
    parser.add_argument("--mechanics-batch-size", type=int, default=1024)
    parser.add_argument("--motoring-efficiency", type=float, default=0.85)
    parser.add_argument("--regen-efficiency", type=float, default=0.60)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--mechanics-correction-phases",
        type=int,
        default=2,
        help="MLP fine-tuning phases after re-relaxing at predicted stiffnesses.",
    )
    parser.add_argument(
        "--mechanics-correction-profiles",
        type=int,
        default=0,
        help="Profiles per refresh; 0 uses all training profiles (default).",
    )
    parser.add_argument(
        "--mechanics-correction-samples",
        type=int,
        default=0,
        help="Samples per profile per refresh; 0 uses every sample (default).",
    )
    parser.add_argument("--mechanics-correction-iterations", type=int, default=1500)
    parser.add_argument("--mechanics-correction-learning-rate", type=float, default=0.001)
    parser.add_argument("--output-name", default="profile_passive_3d_60spring")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    topology = load_spatial_topology(args.topology, torch.device(args.device))
    if args.nonlinear_ratio < 0.0:
        parser.error("--nonlinear-ratio must be nonnegative")
    if args.nonlinear_reference_extension <= 0.0:
        parser.error("--nonlinear-reference-extension must be positive")
    if args.nonlinear_power == 1 and args.nonlinear_ratio:
        parser.error("Use --nonlinear-ratio 0 with linear power 1")
    topology["nonlinear_power"] = args.nonlinear_power
    topology["nonlinear_ratio"] = args.nonlinear_ratio
    topology["nonlinear_reference_extension"] = args.nonlinear_reference_extension
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, args.relaxation_steps)
    rng = np.random.default_rng(args.seed)
    train_profiles = generate_classified_profile_parameters(rng, args.profiles_per_family)
    test_profiles = generate_classified_profile_parameters(rng, args.test_profiles_per_family)
    common = (angles, basis, args.duration, args.samples)
    train = build_profile_dataset(
        train_profiles, *common, args.seed + 20_000
    )
    test = build_profile_dataset(
        test_profiles, *common, args.seed + 30_000
    )
    base_k = topology["initial_stiffness"].detach().cpu().numpy()
    model, history = train_profile_model(
        train,
        base_k,
        args.hidden_dim,
        args.iterations,
        args.learning_rate,
        args.min_stiffness,
        1.0,
        0.0,
        args.energy_weight,
        0.0,
        args.motoring_efficiency,
        args.regen_efficiency,
        args.seed,
        device=args.device,
        progress_interval=args.progress_interval,
        unbounded_stiffness=True,
    )
    correction_rng = np.random.default_rng(args.seed + 70_000)
    for phase in range(1, args.mechanics_correction_phases + 1):
        profile_indices, sample_indices = correction_indices(
            len(train["profile_features"]),
            args.samples,
            args.mechanics_correction_profiles,
            args.mechanics_correction_samples,
            correction_rng,
        )
        correction = subset_profile_dataset(train, profile_indices)
        if len(sample_indices) < args.samples:
            for key in ("theta", "theta_dot", "target", "basis", "t"):
                correction[key] = correction[key][:, sample_indices]
            correction["samples_per_profile"] = len(sample_indices)
        print(
            f"Re-relaxing 3D mechanics at MLP predictions "
            f"({phase}/{args.mechanics_correction_phases}) over "
            f"{len(profile_indices)} profiles x {len(sample_indices)} samples..."
        )
        correction = refresh_mlp_spatial_basis(
            model,
            correction,
            topology,
            args.min_stiffness,
            args.relaxation_steps,
            args.mechanics_batch_size,
        )
        model, phase_history = train_profile_model(
            correction,
            base_k,
            args.hidden_dim,
            args.mechanics_correction_iterations,
            args.mechanics_correction_learning_rate,
            args.min_stiffness,
            1.0,
            0.0,
            args.energy_weight,
            0.0,
            args.motoring_efficiency,
            args.regen_efficiency,
            args.seed + phase,
            device=args.device,
            progress_interval=args.progress_interval,
            initial_model=model,
            unbounded_stiffness=True,
        )
        offset = history[-1][0]
        history.extend(
            (offset + iteration, rmse, loss, energy)
            for iteration, rmse, loss, energy in phase_history
        )
    test_k = predict_profile_stiffness(
        model, test, args.min_stiffness, 1.0, unbounded_stiffness=True
    )
    torque, force_residual = relaxed_spatial_profile_torque(
        test, topology, test_k, args.relaxation_steps, args.mechanics_batch_size
    )
    rows = summary_rows(
        test_profiles, test, torque, test_k,
        args.motoring_efficiency, args.regen_efficiency
    )
    baseline = sum(row["baseline_energy_burden_j"] for row in rows)
    assisted = sum(row["assisted_energy_burden_j"] for row in rows)
    offload = 100.0 * (baseline - assisted) / baseline
    print(f"Held-out exact 3D aggregate offload: {offload:.3f}%")
    print(f"Mean profile offload: {np.mean([r['offload_pct'] for r in rows]):.3f}%")
    print(f"Negative-offload profiles: {100*np.mean([r['offload_pct'] < 0 for r in rows]):.2f}%")
    print(f"Mean internal-force residual: {np.mean(force_residual):.6f} N")
    print(
        f"Predicted stiffness N/m: median {np.median(test_k):.3f} | "
        f"p95 {np.percentile(test_k,95):.3f} | max {np.max(test_k):.3f}"
    )
    model_dir = PROJECT_ROOT / "models" / "profile_conditioned_passive_3d"
    table_dir = PROJECT_ROOT / "tables" / "profile_conditioned_passive_3d"
    save_model(
        model_dir / f"{args.output_name}.npz",
        model,
        args.output_name,
        args.min_stiffness,
        np.inf,
        controller_type="profile_conditioned_passive_3d",
        stiffness_parameterization="positive_unbounded_softplus",
        feature_type="five_knot_torque_angle_profile",
        topology=str(args.topology),
        spring_count=len(base_k),
        profile_classification=PROFILE_CLASSIFICATION,
        profile_families=np.asarray(PROFILE_FAMILIES),
        relaxation_steps=args.relaxation_steps,
        mechanics_correction_phases=args.mechanics_correction_phases,
        mechanics_correction_profiles=args.mechanics_correction_profiles,
        mechanics_correction_samples=args.mechanics_correction_samples,
        mechanics_correction_resolved_profiles=(
            len(train["profile_features"])
            if args.mechanics_correction_profiles == 0
            else min(args.mechanics_correction_profiles, len(train["profile_features"]))
        ),
        mechanics_correction_resolved_samples=(
            args.samples
            if args.mechanics_correction_samples == 0
            else min(args.mechanics_correction_samples, args.samples)
        ),
        nonlinear_power=args.nonlinear_power,
        nonlinear_ratio=args.nonlinear_ratio,
        nonlinear_reference_extension=args.nonlinear_reference_extension,
        seed=args.seed,
    )
    write_rows(table_dir / f"{args.output_name}_test_results.csv", rows)
    write_stiffness_rows(
        table_dir / f"{args.output_name}_test_stiffness.csv", test_profiles, test_k
    )
    write_rows(
        table_dir / f"{args.output_name}_training_history.csv",
        [
            {"iteration": i, "residual_rmse_nm": r, "loss": loss, "energy_ratio": energy}
            for i, r, loss, energy in history
        ],
    )


if __name__ == "__main__":
    main()
