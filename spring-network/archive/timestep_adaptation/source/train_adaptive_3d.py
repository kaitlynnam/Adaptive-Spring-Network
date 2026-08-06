"""Train a causal adaptive-stiffness controller on genuine spatial mechanics."""

from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES, forward, save_model
from energy_accounting import validate_efficiencies
from mechanics_3d import (
    load_spatial_topology,
    torque_and_residual,
    torque_components,
    torque_components_and_residual,
)
from profile_generator import (
    PROFILE_FAMILIES,
    generate_classified_profile_parameters,
)
from train_adaptive_dataset import (
    aggregate_profile_rows,
    build_dataset,
    normalization_scales,
    plot_training_convergence,
    print_model_comparison,
    print_summary,
    print_worst_cases,
    summarize_profiles,
    train_model,
    write_model_comparison_rows,
    write_profile_rows,
)


DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial" / "internal_fan_3d_48_spring_densest.json"
)


def initial_basis(topology, angles, relaxation_steps):
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(angles), 1)
    components = torque_components(
        topology,
        torch.as_tensor(angles, dtype=torch.float32, device=stiffness.device),
        stiffness,
        relaxation_steps,
    )
    return (
        components / torch.clamp(stiffness, min=1e-9)
    ).detach().cpu().numpy()


def causal_spatial_rollout(
    model, dataset, topology, min_k, max_k, relaxation_steps, return_basis=False
):
    samples = dataset["samples_per_profile"]
    profiles = len(dataset["target"]) // samples
    window = dataset["window_size"]
    torque_scale = max(dataset["torque_scale"], 1e-9)
    motion = dataset["features"].reshape(profiles, samples, -1)
    target = dataset["target"].reshape(profiles, samples)
    theta = dataset["theta"].reshape(profiles, samples)
    history = np.zeros((profiles, window, 3), dtype=float)
    predicted = np.empty((profiles, samples), dtype=float)
    stiffness_schedule = np.empty(
        (profiles, samples, len(topology["spring_a"])), dtype=float
    )
    basis = np.empty_like(stiffness_schedule) if return_basis else None
    residual = np.empty((profiles, samples), dtype=float)

    for sample_index in range(samples):
        inputs = np.hstack((motion[:, sample_index, :], history.reshape(profiles, -1)))
        stiffness, _ = forward(model, inputs, min_k, max_k)
        theta_tensor = torch.as_tensor(
            theta[:, sample_index],
            dtype=torch.float32,
            device=topology["local_positions"].device,
        )
        stiffness_tensor = torch.as_tensor(
            stiffness,
            dtype=torch.float32,
            device=topology["local_positions"].device,
        )
        if return_basis:
            components, force_residual, _ = torque_components_and_residual(
                topology, theta_tensor, stiffness_tensor, relaxation_steps
            )
            torque = torch.sum(components, dim=1)
            basis[:, sample_index, :] = (
                components / torch.clamp(stiffness_tensor, min=1e-9)
            ).detach().cpu().numpy()
        else:
            torque, force_residual, _ = torque_and_residual(
                topology, theta_tensor, stiffness_tensor, relaxation_steps
            )
        torque_np = torque.detach().cpu().numpy()
        predicted[:, sample_index] = torque_np
        stiffness_schedule[:, sample_index, :] = stiffness
        residual[:, sample_index] = force_residual.detach().cpu().numpy()
        motor = target[:, sample_index] - torque_np
        realized = np.stack(
            (target[:, sample_index], torque_np, motor), axis=1
        ) / torque_scale
        history = np.concatenate((history[:, 1:, :], realized[:, None, :]), axis=1)
    outputs = (
        predicted.reshape(-1),
        stiffness_schedule.reshape(-1, stiffness_schedule.shape[2]),
        residual.reshape(-1),
    )
    if return_basis:
        return (*outputs, basis.reshape(-1, basis.shape[2]))
    return outputs


def fixed_spatial_torque(
    dataset, topology, stiffness, relaxation_steps, batch_size=4096
):
    torque_chunks, residual_chunks = [], []
    device = topology["local_positions"].device
    for start in range(0, len(dataset["theta"]), batch_size):
        theta = torch.as_tensor(
            dataset["theta"][start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        schedule = torch.as_tensor(
            np.repeat(np.asarray(stiffness)[None, :], len(theta), axis=0),
            dtype=torch.float32,
            device=device,
        )
        torque, residual, _ = torque_and_residual(
            topology, theta, schedule, relaxation_steps
        )
        torque_chunks.append(torque.detach().cpu().numpy())
        residual_chunks.append(residual.detach().cpu().numpy())
    return np.concatenate(torque_chunks), np.concatenate(residual_chunks)


def plot_examples(path, params, dataset, predicted):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    samples = dataset["samples_per_profile"]
    for index, axis in enumerate(axes.flat):
        if index >= len(params):
            axis.set_visible(False)
            continue
        start, stop = index * samples, (index + 1) * samples
        target = dataset["target"][start:stop]
        spring = predicted[start:stop]
        residual = target - spring
        angle = np.rad2deg(dataset["theta"][start:stop])
        axis.scatter(angle, spring, s=13, alpha=0.72, label="3D spring torque")
        axis.scatter(angle, residual, s=12, marker="x", alpha=0.6, label="residual motor")
        order = np.argsort(angle)
        axis.plot(angle[order], target[order], "k--", linewidth=1.7, label="target")
        axis.set_title(f"{params[index]['family']} / {params[index]['name']}")
        axis.set_xlabel("joint angle [deg]")
        axis.set_ylabel("joint-axis torque [Nm]")
        axis.grid(True, alpha=0.25)
    axes.flat[0].legend()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--profiles-per-family", type=int, default=2000)
    parser.add_argument("--test-profiles-per-family", type=int, default=400)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--min-stiffness", type=float, default=1.0)
    parser.add_argument("--max-stiffness", type=float, default=800.0)
    parser.add_argument("--surrogate-refreshes", type=int, default=2)
    parser.add_argument("--relaxation-steps", type=int, default=160)
    parser.add_argument(
        "--evaluation-relaxation-steps", type=int, default=None,
        help="Deeper relaxation for fixed baselines and final reported rollouts.",
    )
    parser.add_argument("--mechanics-batch-size", type=int, default=4096)
    parser.add_argument("--motion-mode", choices=["triangular", "randomized"], default="triangular")
    parser.add_argument("--fixed-frequency-hz", type=float, default=1.0)
    parser.add_argument("--motoring-efficiency", type=float, default=1.0)
    parser.add_argument("--regen-efficiency", type=float, default=0.0)
    parser.add_argument(
        "--cubic-ratio", type=float, default=0.0,
        help="Cubic hardening ratio at the reference extension (0 = linear).",
    )
    parser.add_argument(
        "--cubic-reference-extension", type=float, default=0.6,
        help="Reference extension in metres for --cubic-ratio.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--output-name", default="spatial_internal_fan_3d_5000")
    args = parser.parse_args()
    evaluation_relaxation_steps = (
        args.evaluation_relaxation_steps
        if args.evaluation_relaxation_steps is not None
        else args.relaxation_steps
    )
    validate_efficiencies(args.motoring_efficiency, args.regen_efficiency)
    if args.surrogate_refreshes >= args.iterations:
        parser.error("--surrogate-refreshes must be smaller than --iterations")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    topology = load_spatial_topology(args.topology, device)
    topology["cubic_ratio"] = args.cubic_ratio
    topology["cubic_reference_extension"] = args.cubic_reference_extension
    base_k = topology["initial_stiffness"].detach().cpu().numpy()
    angles = np.radians(ANGLE_DEGREES)
    basis = initial_basis(topology, angles, args.relaxation_steps)

    rng = np.random.default_rng(args.seed)
    train_params = generate_classified_profile_parameters(rng, args.profiles_per_family)
    test_params = generate_classified_profile_parameters(rng, args.test_profiles_per_family)
    scales = normalization_scales(
        train_params,
        args.duration,
        args.samples,
        args.seed + 10_000,
        args.window_size,
        progress_interval=args.progress_interval,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
    )
    common = dict(
        angles_rad=angles,
        basis_by_angle=basis,
        duration=args.duration,
        samples=args.samples,
        window_size=args.window_size,
        scales=scales,
        stiffness_update_mode="timestep",
        progress_interval=args.progress_interval,
        include_profile_descriptor=False,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
    )
    train = build_dataset(
        train_params, seed=args.seed + 20_000,
        progress_label="3D training profiles", **common
    )
    test = build_dataset(
        test_params, seed=args.seed + 30_000,
        progress_label="3D test profiles", **common
    )
    baseline_train, baseline_train_residual = fixed_spatial_torque(
        train, topology, base_k, evaluation_relaxation_steps,
        args.mechanics_batch_size
    )
    baseline_test, baseline_test_residual = fixed_spatial_torque(
        test, topology, base_k, evaluation_relaxation_steps,
        args.mechanics_batch_size
    )
    train_baseline_rmse = float(
        np.sqrt(np.mean((baseline_train - train["target"]) ** 2))
    )
    test_baseline_rmse = float(
        np.sqrt(np.mean((baseline_test - test["target"]) ** 2))
    )
    print(
        f"Spatial topology: {topology['data']['name']} | "
        f"{len(topology['names'])} nodes | {len(base_k)} springs | "
        f"{len(topology['internal_indices'])} free 3D internal nodes"
    )
    print(
        f"Fixed baseline RMSE: train {train_baseline_rmse:.3f} Nm | "
        f"test {test_baseline_rmse:.3f} Nm"
    )
    print(
        f"Baseline equilibrium residual: train max {np.max(baseline_train_residual):.5f} N | "
        f"test max {np.max(baseline_test_residual):.5f} N"
    )

    phase_count = args.surrogate_refreshes + 1
    counts = [
        args.iterations // phase_count
        + (1 if i < args.iterations % phase_count else 0)
        for i in range(phase_count)
    ]
    model, combined_history, completed = None, None, 0
    active_train = train
    for phase, count in enumerate(counts):
        if phase:
            print(f"Refreshing spatial surrogate ({phase}/{args.surrogate_refreshes})...")
            _, _, refresh_residual, refreshed_basis = causal_spatial_rollout(
                model,
                train,
                topology,
                args.min_stiffness,
                args.max_stiffness,
                args.relaxation_steps,
                return_basis=True,
            )
            active_train = dict(train)
            active_train["basis"] = refreshed_basis
            print(
                f"Refresh equilibrium residual: mean {np.mean(refresh_residual):.5f} N | "
                f"max {np.max(refresh_residual):.5f} N"
            )
        model, history = train_model(
            active_train,
            base_k,
            args.hidden_dim,
            count,
            args.learning_rate,
            args.min_stiffness,
            args.max_stiffness,
            0.0,
            args.seed,
            args.progress_interval,
            device=str(device),
            energy_weight=0.0,
            motoring_efficiency=args.motoring_efficiency,
            regen_efficiency=args.regen_efficiency,
            stiffness_change_weight=0.0,
            optimizer_name="adam",
            initial_model=model,
        )
        history["iteration"] = [completed + value for value in history["iteration"]]
        completed += count
        if combined_history is None:
            combined_history = history
        else:
            for key, values in history.items():
                combined_history[key].extend(values)

    train_pred, train_stiffness, train_residual = causal_spatial_rollout(
        model, train, topology, args.min_stiffness, args.max_stiffness,
        evaluation_relaxation_steps
    )
    test_pred, test_stiffness, test_residual = causal_spatial_rollout(
        model, test, topology, args.min_stiffness, args.max_stiffness,
        evaluation_relaxation_steps
    )
    energy_args = (args.motoring_efficiency, args.regen_efficiency)
    train_rows = summarize_profiles(train_params, train, train_pred, *energy_args)
    test_rows = summarize_profiles(test_params, test, test_pred, *energy_args)
    baseline_rows = summarize_profiles(test_params, test, baseline_test, *energy_args)
    comparison = [
        aggregate_profile_rows("fixed_spatial_baseline", "relaxed_3d", baseline_rows),
        aggregate_profile_rows("adaptive_spatial", "relaxed_3d", test_rows),
    ]
    print_summary("3D training performance", train_rows)
    print_summary("3D held-out performance", test_rows)
    print_worst_cases(test_rows)
    print_model_comparison(comparison)
    print(
        f"Final equilibrium residual: train mean/max "
        f"{np.mean(train_residual):.5f}/{np.max(train_residual):.5f} N | "
        f"test mean/max {np.mean(test_residual):.5f}/{np.max(test_residual):.5f} N"
    )

    model_path = PROJECT_ROOT / "models" / "spatial" / f"{args.output_name}.npz"
    table_dir = PROJECT_ROOT / "tables" / "spatial"
    plot_dir = PROJECT_ROOT / "plots" / "runs" / "spatial"
    save_model(
        model_path,
        model,
        args.output_name,
        args.min_stiffness,
        args.max_stiffness,
        feature_type="causal_motion_torque_window_3d",
        topology=str(args.topology),
        spatial_mechanics=True,
        joint_axis=topology["data"]["joint_axis"],
        node_count=len(topology["names"]),
        spring_count=len(base_k),
        internal_node_count=len(topology["internal_indices"]),
        profiles_per_family=args.profiles_per_family,
        test_profiles_per_family=args.test_profiles_per_family,
        duration=args.duration,
        samples=args.samples,
        window_size=args.window_size,
        theta_scale=scales["theta"],
        theta_dot_scale=scales["theta_dot"],
        theta_ddot_scale=scales["theta_ddot"],
        torque_scale=scales["torque"],
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
        relaxation_steps=args.relaxation_steps,
        evaluation_relaxation_steps=evaluation_relaxation_steps,
        cubic_ratio=args.cubic_ratio,
        cubic_reference_extension=args.cubic_reference_extension,
        mechanics_batch_size=args.mechanics_batch_size,
        surrogate_refreshes=args.surrogate_refreshes,
        seed=args.seed,
    )
    write_profile_rows(table_dir / f"{args.output_name}_train_results.csv", train_rows)
    write_profile_rows(table_dir / f"{args.output_name}_test_results.csv", test_rows)
    write_model_comparison_rows(
        table_dir / f"{args.output_name}_mechanics_comparison.csv", comparison
    )
    plot_training_convergence(
        plot_dir / f"{args.output_name}_training_convergence.png",
        combined_history,
        train_baseline_rmse,
        test_baseline_rmse,
    )
    plot_examples(
        plot_dir / f"{args.output_name}_test_examples.png",
        test_params,
        test,
        test_pred,
    )
    np.savetxt(
        table_dir / f"{args.output_name}_equilibrium_residual_summary.csv",
        np.asarray([
            [np.mean(train_residual), np.max(train_residual)],
            [np.mean(test_residual), np.max(test_residual)],
        ]),
        delimiter=",",
        header="mean_force_residual_n,max_force_residual_n",
        comments="",
    )
    print(f"Saved spatial checkpoint: {model_path}")


if __name__ == "__main__":
    main()
