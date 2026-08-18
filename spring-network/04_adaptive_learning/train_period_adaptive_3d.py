"""Train a causal, once-per-period adaptive-stiffness controller in 3D.

Period zero always uses the topology's initial stiffness.  At each later
boundary the MLP consumes exactly the preceding period's measured motion and
torques and chooses one stiffness vector, which is held for the whole period.
The loss is evaluated on the period after the observation.
"""

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

from adaptive_model import ANGLE_DEGREES, save_model
from mechanics_3d import load_spatial_topology, torque_and_residual, torque_components
from period_adaptive_support import (
    figure_path, generate_motion_trajectory, initialize_period_model, interpolate_basis,
    spatial_initial_basis,
)
from profile_generator import DEFAULT_TORQUE_LIMIT_NM, generate_profile_parameters

DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"
)


def build_period_dataset(profiles, angles, angle_basis, period_seconds,
                         samples_per_period, seed, motion_mode="triangular",
                         frequency_hz=None, torque_scale=DEFAULT_TORQUE_LIMIT_NM):
    """Build one-period trajectories; these are repeated during causal rollout."""
    frequency_hz = (1.0 / period_seconds) if frequency_hz is None else frequency_hz
    if not np.isclose(frequency_hz * period_seconds, 1.0, rtol=1e-6, atol=1e-9):
        raise ValueError("period_seconds must equal one motion period (frequency * period = 1)")
    rows = {key: [] for key in ("theta", "theta_dot", "theta_ddot", "target", "basis", "t")}
    for index, profile in enumerate(profiles):
        t, theta, theta_dot, theta_ddot, target = generate_motion_trajectory(
            profile, period_seconds, samples_per_period, seed + index,
            motion_mode=motion_mode, fixed_frequency_hz=frequency_hz,
        )
        rows["t"].append(t)
        rows["theta"].append(theta)
        rows["theta_dot"].append(theta_dot)
        rows["theta_ddot"].append(theta_ddot)
        rows["target"].append(target)
        rows["basis"].append(interpolate_basis(angle_basis, angles, theta))
    result = {key: np.asarray(value, dtype=float) for key, value in rows.items()}
    result.update(samples_per_period=int(samples_per_period), period_seconds=float(period_seconds),
                  frequency_hz=float(frequency_hz), torque_scale=float(torque_scale))
    return result


def period_observation(theta, theta_dot, theta_ddot, target, spring, motor,
                       motion_scales, torque_scale):
    """Flatten one complete period in time order into the controller input."""
    channels = torch.stack((
        theta / motion_scales[0], theta_dot / motion_scales[1],
        theta_ddot / motion_scales[2], target / torque_scale,
        spring / torque_scale, motor / torque_scale,
    ), dim=2)
    return channels.reshape(channels.shape[0], -1)


def rollout(parameters, dataset, initial_k, min_k, periods, detach_buffer=True,
            stiffness_lower=None, stiffness_upper=None):
    """Run causal periods; no network call is made for the first period."""
    basis, target = dataset["basis"], dataset["target"]
    period_basis = dataset.get("period_basis")
    if period_basis is not None and period_basis.shape[1] < periods:
        raise ValueError("period-specific basis has fewer periods than the rollout")
    theta, theta_dot, theta_ddot = dataset["theta"], dataset["theta_dot"], dataset["theta_ddot"]
    batch = target.shape[0]
    stiffness = (
        initial_k.unsqueeze(0).expand(batch, -1)
        if initial_k.ndim == 1 else initial_k
    )
    if stiffness.shape[0] != batch:
        raise ValueError("initial stiffness batch must match trajectory batch")
    stiffness_periods, torque_periods = [], []
    motion_scales = dataset["motion_scales"]
    for period_index in range(periods):
        active_basis = basis if period_basis is None else period_basis[:, period_index]
        spring = torch.sum(active_basis * stiffness[:, None, :], dim=2)
        motor = target - spring
        stiffness_periods.append(stiffness)
        torque_periods.append(spring)
        if period_index + 1 < periods:
            observation = period_observation(
                theta, theta_dot, theta_ddot, target, spring, motor,
                motion_scales, dataset["torque_scale"],
            )
            if detach_buffer:
                observation = observation.detach()
            hidden = torch.tanh(observation @ parameters["w1"] + parameters["b1"])
            logits = hidden @ parameters["w2"] + parameters["b2"]
            if stiffness_lower is None or stiffness_upper is None:
                stiffness = min_k + torch.nn.functional.softplus(logits)
            else:
                stiffness = torch.clamp(
                    min_k + torch.nn.functional.softplus(logits),
                    min=stiffness_lower, max=stiffness_upper,
                )
    return torch.stack(torque_periods, dim=1), torch.stack(stiffness_periods, dim=1)


def tensor_dataset(dataset, device):
    result = {}
    for key in ("theta", "theta_dot", "theta_ddot", "target", "basis"):
        result[key] = torch.as_tensor(dataset[key], dtype=torch.float32, device=device)
    if "period_basis" in dataset:
        result["period_basis"] = torch.as_tensor(
            dataset["period_basis"], dtype=torch.float32, device=device
        )
    scales = np.array([
        max(np.percentile(np.abs(dataset["theta"]), 95), np.deg2rad(1)),
        max(np.percentile(np.abs(dataset["theta_dot"]), 95), 0.1),
        max(np.percentile(np.abs(dataset["theta_ddot"]), 95), 0.5),
    ])
    result["motion_scales"] = torch.as_tensor(scales, dtype=torch.float32, device=device)
    result["torque_scale"] = max(float(dataset["torque_scale"]), 1e-9)
    return result, scales


def train_period_model(dataset, initial_k, hidden_dim=256, iterations=5000,
                       learning_rate=1e-3, min_k=0.0, periods=2, seed=101,
                       device="cpu", progress_interval=100, initial_model=None,
                       observation_variants=4, observation_stiffness_log_std=0.7,
                       training_mode="closed_loop", training_periods=6,
                       initial_stiffness_log_std=0.7, stiffness_lower=None,
                       stiffness_upper=None, stiffness_order_weight=0.1,
                       objective="torque_mse"):
    """Learn completed-period data -> stiffness that fits that same period.

    The first-period/default policy belongs to deployment and is deliberately
    In closed-loop mode, each trajectory starts from a default or randomized
    stiffness and all later periods use the network's own preceding output.
    """
    torch_device = torch.device(device)
    data, scales = tensor_dataset(dataset, torch_device)
    initial_k = np.asarray(initial_k, dtype=float)
    input_dim = dataset["samples_per_period"] * 6
    model = (
        initialize_period_model(
            np.random.default_rng(seed), input_dim, hidden_dim,
            dataset["basis"].shape[2], initial_k, min_k,
        )
        if initial_model is None
        else {name: np.asarray(value, dtype=float).copy()
              for name, value in initial_model.items()}
    )
    if model["w1"].shape != (input_dim, hidden_dim):
        raise ValueError(
            f"checkpoint input/hidden shape {model['w1'].shape} does not match "
            f"requested {(input_dim, hidden_dim)}"
        )
    if model["w2"].shape[1] != dataset["basis"].shape[2]:
        raise ValueError("checkpoint spring count does not match the active topology")
    parameters = {name: torch.tensor(value, dtype=torch.float32, device=torch_device,
                                     requires_grad=True) for name, value in model.items()}
    base = torch.as_tensor(initial_k, dtype=torch.float32, device=torch_device)
    lower_tensor = (None if stiffness_lower is None else
                    torch.as_tensor(stiffness_lower, dtype=torch.float32, device=torch_device))
    upper_tensor = (None if stiffness_upper is None else
                    torch.as_tensor(stiffness_upper, dtype=torch.float32, device=torch_device))
    if training_mode not in {"closed_loop", "independent"}:
        raise ValueError("training_mode must be closed_loop or independent")
    if training_periods < 2:
        raise ValueError("closed-loop training requires at least two periods")
    if observation_variants < 1:
        raise ValueError("observation_variants must be at least one")
    if objective not in {"torque_mse", "motor_work"}:
        raise ValueError("objective must be torque_mse or motor_work")
    profile_count, sample_count, spring_count = data["basis"].shape
    generator = torch.Generator(device=torch_device).manual_seed(seed + 31_337)
    variant_count = observation_variants if training_mode == "independent" else 1
    observation_k = base[None, None, :].expand(profile_count, variant_count, -1).clone()
    if variant_count > 1:
        noise = torch.randn(
            (profile_count, variant_count - 1, spring_count),
            generator=generator, dtype=torch.float32, device=torch_device,
        )
        observation_k[:, 1:, :] = torch.clamp(
            base[None, None, :] * torch.exp(observation_stiffness_log_std * noise),
            min=(lower_tensor[None, None, :] if lower_tensor is not None else max(min_k, 1e-6)),
            max=(upper_tensor[None, None, :] if upper_tensor is not None else None),
        )
    observed_spring = torch.sum(
        data["basis"][:, None, :, :] * observation_k[:, :, None, :], dim=3
    ).reshape(profile_count * variant_count, sample_count)
    expanded = {
        key: value[:, None, :].expand(-1, variant_count, -1).reshape(
            profile_count * variant_count, sample_count
        )
        for key, value in (("theta", data["theta"]), ("theta_dot", data["theta_dot"]),
                           ("theta_ddot", data["theta_ddot"]), ("target", data["target"]))
    }
    observation = period_observation(
        expanded["theta"], expanded["theta_dot"], expanded["theta_ddot"],
        expanded["target"], observed_spring, expanded["target"] - observed_spring,
        data["motion_scales"], data["torque_scale"],
    )
    expanded_basis = data["basis"][:, None, :, :].expand(
        -1, variant_count, -1, -1
    ).reshape(profile_count * variant_count, sample_count, spring_count)
    rollout_initial_k = base[None, :].expand(profile_count, -1).clone()
    if training_mode == "closed_loop" and profile_count > 1:
        # Preserve default starts for half the batch and broaden the other half.
        random_count = profile_count // 2
        start_noise = torch.randn(
            (random_count, spring_count), generator=generator,
            dtype=torch.float32, device=torch_device,
        )
        rollout_initial_k[-random_count:] = torch.clamp(
            base[None, :] * torch.exp(initial_stiffness_log_std * start_noise),
            min=(lower_tensor[None, :] if lower_tensor is not None else max(min_k, 1e-6)),
            max=(upper_tensor[None, :] if upper_tensor is not None else None),
        )
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    best_loss, best_model, history = float("inf"), None, []
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        if training_mode == "closed_loop":
            torque, rollout_stiffness = rollout(
                parameters, data, rollout_initial_k, min_k, training_periods,
                detach_buffer=True, stiffness_lower=lower_tensor,
                stiffness_upper=upper_tensor,
            )
            residual = torque[:, 1:, :] - data["target"][:, None, :]
            torque_mse = torch.mean(residual**2)
            primary_loss = (
                torque_mse if objective == "torque_mse" else
                torch.mean(torch.abs(residual * data["theta_dot"][:, None, :]))
            )
            order_change = torch.log10(
                torch.clamp(rollout_stiffness[:, 1:, :], min=1e-9)
                / torch.clamp(base[None, None, :], min=1e-9)
            )
            loss = primary_loss + stiffness_order_weight * primary_loss.detach() * torch.mean(
                order_change**2
            )
        else:
            hidden = torch.tanh(observation @ parameters["w1"] + parameters["b1"])
            logits = hidden @ parameters["w2"] + parameters["b2"]
            if lower_tensor is not None:
                stiffness = torch.clamp(
                    min_k + torch.nn.functional.softplus(logits),
                    min=lower_tensor, max=upper_tensor,
                )
            else:
                stiffness = min_k + torch.nn.functional.softplus(logits)
            torque = torch.sum(expanded_basis * stiffness[:, None, :], dim=2)
            residual = torque - expanded["target"]
            torque_mse = torch.mean(residual**2)
            primary_loss = (
                torque_mse if objective == "torque_mse" else
                torch.mean(torch.abs(residual * expanded["theta_dot"]))
            )
            loss = primary_loss
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        rmse = float(torch.sqrt(torque_mse.detach()).cpu())
        history.append((iteration, rmse, value))
        if value < best_loss:
            best_loss = value
            best_model = {name: tensor.detach().cpu().numpy().copy()
                          for name, tensor in parameters.items()}
        if progress_interval and (iteration == 1 or iteration == iterations
                                  or iteration % progress_interval == 0):
            label = "closed-loop adapted" if training_mode == "closed_loop" else "period-fit"
            suffix = (
                f"RMSE {rmse:8.3f} N*m" if objective == "torque_mse" else
                f"motor-work proxy {value:8.3f} W | RMSE {rmse:8.3f} N*m"
            )
            print(f"iteration {iteration:5d} | {label} {suffix}")
    return best_model, history, scales


def predict_period_schedule(model, dataset, initial_k, min_k=0.0, periods=2,
                            device="cpu", motion_scales=None,
                            rollout_initial_stiffness=None, stiffness_lower=None,
                            stiffness_upper=None):
    """Inference helper returning [trajectory, period, sample/spring] arrays."""
    data, inferred_scales = tensor_dataset(dataset, torch.device(device))
    scales = inferred_scales if motion_scales is None else np.asarray(motion_scales)
    data["motion_scales"] = torch.as_tensor(scales, dtype=torch.float32, device=device)
    params = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in model.items()}
    base = torch.as_tensor(
        initial_k if rollout_initial_stiffness is None else rollout_initial_stiffness,
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        lower = (None if stiffness_lower is None else
                 torch.as_tensor(stiffness_lower, dtype=torch.float32, device=device))
        upper = (None if stiffness_upper is None else
                 torch.as_tensor(stiffness_upper, dtype=torch.float32, device=device))
        torque, stiffness = rollout(
            params, data, base, min_k, periods,
            stiffness_lower=lower, stiffness_upper=upper,
        )
    return torque.cpu().numpy(), stiffness.cpu().numpy()


def refresh_period_basis(model, dataset, topology, initial_k, min_k,
                         motion_scales, relaxation_steps, rollout_periods=2,
                         batch_size=1024, progress_interval=10,
                         random_initial_log_std=0.0, seed=0,
                         stiffness_lower=None, stiffness_upper=None):
    """Rebuild a separate local torque basis for every rollout period."""
    rollout_initial = np.broadcast_to(
        np.asarray(initial_k, dtype=float)[None, :],
        (len(dataset["target"]), len(initial_k)),
    ).copy()
    if random_initial_log_std > 0 and len(rollout_initial) > 1:
        random_count = len(rollout_initial) // 2
        noise = np.random.default_rng(seed).normal(
            size=(random_count, rollout_initial.shape[1])
        )
        rollout_initial[-random_count:] = np.maximum(
            min_k, rollout_initial[-random_count:] * np.exp(random_initial_log_std * noise)
        )
        if stiffness_lower is not None:
            rollout_initial[-random_count:] = np.maximum(
                rollout_initial[-random_count:], np.asarray(stiffness_lower)[None, :]
            )
        if stiffness_upper is not None:
            rollout_initial[-random_count:] = np.minimum(
                rollout_initial[-random_count:], np.asarray(stiffness_upper)[None, :]
            )
    _, schedule = predict_period_schedule(
        model, dataset, initial_k, min_k, periods=rollout_periods,
        device=str(topology["local_positions"].device), motion_scales=motion_scales,
        rollout_initial_stiffness=rollout_initial,
        stiffness_lower=stiffness_lower, stiffness_upper=stiffness_upper,
    )
    profiles, samples, spring_count = dataset["basis"].shape
    stiffness = np.broadcast_to(
        schedule[:, :, None, :],
        (profiles, rollout_periods, samples, spring_count),
    ).reshape(-1, spring_count)
    theta = np.broadcast_to(
        dataset["theta"][:, None, :], (profiles, rollout_periods, samples)
    ).reshape(-1)
    components = np.empty_like(stiffness)
    device = topology["local_positions"].device
    batches = int(np.ceil(len(theta) / batch_size))
    for batch_index, start in enumerate(range(0, len(theta), batch_size), start=1):
        stop = min(start + batch_size, len(theta))
        values = torque_components(
            topology,
            torch.as_tensor(theta[start:stop].copy(), dtype=torch.float32, device=device),
            torch.as_tensor(stiffness[start:stop].copy(), dtype=torch.float32, device=device),
            relaxation_steps,
        )
        components[start:stop] = values.detach().cpu().numpy()
        if progress_interval and (batch_index == 1 or batch_index == batches
                                  or batch_index % progress_interval == 0):
            print(f"Mechanics refresh: {batch_index}/{batches} batches", flush=True)
    refreshed = dict(dataset)
    refreshed["period_basis"] = (
        components / np.maximum(stiffness, 1e-6)
    ).reshape(profiles, rollout_periods, samples, spring_count)
    # Keep a representative basis for independent mode and legacy helpers.
    refreshed["basis"] = refreshed["period_basis"][:, -1]
    return refreshed


def exact_period_torque(dataset, topology, stiffness, relaxation_steps,
                        batch_size=1024, progress_interval=10):
    """Evaluate a period-level stiffness schedule with relaxed 3D mechanics."""
    profiles, periods, spring_count = stiffness.shape
    samples = dataset["samples_per_period"]
    schedule = np.broadcast_to(
        stiffness[:, :, None, :], (profiles, periods, samples, spring_count)
    ).reshape(-1, spring_count)
    theta = np.broadcast_to(
        dataset["theta"][:, None, :], (profiles, periods, samples)
    ).reshape(-1)
    torque = np.empty(len(theta), dtype=float)
    residual = np.empty(len(theta), dtype=float)
    device = topology["local_positions"].device
    batches = int(np.ceil(len(theta) / batch_size))
    for batch_index, start in enumerate(range(0, len(theta), batch_size), start=1):
        stop = min(start + batch_size, len(theta))
        values, force_residual, _ = torque_and_residual(
            topology,
            torch.as_tensor(theta[start:stop].copy(), dtype=torch.float32, device=device),
            torch.as_tensor(schedule[start:stop].copy(), dtype=torch.float32, device=device),
            relaxation_steps,
        )
        torque[start:stop] = values.detach().cpu().numpy()
        residual[start:stop] = force_residual.detach().cpu().numpy()
        if progress_interval and (batch_index == 1 or batch_index == batches
                                  or batch_index % progress_interval == 0):
            print(f"Exact held-out mechanics: {batch_index}/{batches} batches")
    shape = (profiles, periods, samples)
    return torque.reshape(shape), residual.reshape(shape)


def save_period_figures(output_dir, name, dataset, torque, stiffness, history,
                        example_count=3):
    """Write the training-convergence figure for the main paper set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iterations = np.asarray([row[0] for row in history])
    rmse = np.asarray([row[1] for row in history])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(iterations, rmse, color="#355c9a", linewidth=2)
    ax.set(xlabel="Training iteration", ylabel="Next-period RMSE [N m]",
           title="Period-adaptive training convergence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path(output_dir, name, "fig07_training_convergence.png"), dpi=180)
    plt.close(fig)


def write_evaluation_summary(path, first_rmse, adapted_rmse, force_residual):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "first_period_default_rmse_nm", "adapted_period_rmse_nm",
            "mean_force_residual_n", "max_force_residual_n",
        ])
        writer.writeheader()
        writer.writerow({
            "first_period_default_rmse_nm": first_rmse,
            "adapted_period_rmse_nm": adapted_rmse,
            "mean_force_residual_n": float(np.mean(force_residual)),
            "max_force_residual_n": float(np.max(force_residual)),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--training-profiles", type=int, default=6000)
    parser.add_argument("--test-profiles", type=int, default=1200)
    parser.add_argument("--period-seconds", type=float, default=5.0)
    parser.add_argument("--samples-per-period", type=int, default=160)
    parser.add_argument("--periods", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-stiffness", type=float, default=0.0)
    parser.add_argument("--min-stiffness-order-change", type=float, default=-0.3)
    parser.add_argument("--max-stiffness-order-change", type=float, default=2.0)
    parser.add_argument(
        "--stiffness-order-weight", type=float, default=0.1,
        help="Soft penalty on squared log10 stiffness change from topology values.",
    )
    parser.add_argument(
        "--objective", choices=["torque_mse", "motor_work"], default="torque_mse",
        help=("Training objective. motor_work minimizes mean absolute residual "
              "torque times angular velocity, matching the benchmark numerator."),
    )
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--mechanics-batch-size", type=int, default=1024)
    parser.add_argument("--mechanics-progress-interval", type=int, default=10)
    parser.add_argument("--figure-examples", type=int, default=3)
    parser.add_argument("--observation-variants", type=int, default=4)
    parser.add_argument("--observation-stiffness-log-std", type=float, default=0.7)
    parser.add_argument(
        "--training-mode", choices=["closed_loop", "independent"], default="closed_loop"
    )
    parser.add_argument("--training-periods", type=int, default=6)
    parser.add_argument("--initial-stiffness-log-std", type=float, default=0.7)
    parser.add_argument(
        "--mechanics-refreshes", type=int, default=0,
        help="Exact relaxed-mechanics basis rebuilds between training phases.",
    )
    parser.add_argument("--motion-mode", choices=["triangular", "randomized"], default="triangular")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--output-name", default="period_adaptive_3d_60spring_bounded_extended")
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="Continue training from an existing period-adaptive .npz checkpoint.",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.periods < 2 or args.samples_per_period < 2 or args.period_seconds <= 0:
        parser.error("periods >= 2, samples-per-period >= 2, and positive period-seconds required")
    if args.observation_variants < 1 or args.observation_stiffness_log_std < 0:
        parser.error("observation variants must be >= 1 and log std must be nonnegative")
    if args.training_periods < 2 or args.initial_stiffness_log_std < 0:
        parser.error("training periods must be >= 2 and initial log std nonnegative")
    if args.min_stiffness_order_change >= args.max_stiffness_order_change:
        parser.error("minimum stiffness order change must be below maximum")
    if args.stiffness_order_weight < 0:
        parser.error("stiffness order weight must be nonnegative")
    if args.mechanics_refreshes < 0 or args.mechanics_refreshes >= args.iterations:
        parser.error("mechanics refreshes must be nonnegative and less than iterations")
    print(
        f"Starting {args.training_mode} training: {args.training_profiles} profiles, "
        f"{args.training_periods} periods, {args.iterations} iterations, "
        f"{args.mechanics_refreshes} mechanics refreshes on {args.device}",
        flush=True,
    )
    device = torch.device(args.device)
    print("Loading 3D topology and building the initial relaxed torque basis...", flush=True)
    topology = load_spatial_topology(args.topology, device)
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, args.relaxation_steps)
    print("Initial relaxed torque basis complete; generating trajectories...", flush=True)
    rng = np.random.default_rng(args.seed)
    frequency = 1.0 / args.period_seconds
    common = (angles, basis, args.period_seconds, args.samples_per_period)
    train = build_period_dataset(generate_profile_parameters(rng, args.training_profiles),
                                 *common, args.seed + 20_000, args.motion_mode, frequency)
    test = build_period_dataset(generate_profile_parameters(rng, args.test_profiles),
                                *common, args.seed + 30_000, args.motion_mode, frequency)
    print("Trajectory generation complete; beginning optimization...", flush=True)
    base_k = topology["initial_stiffness"].detach().cpu().numpy()
    stiffness_lower = base_k * (10.0 ** args.min_stiffness_order_change)
    stiffness_upper = base_k * (10.0 ** args.max_stiffness_order_change)
    initial_model = None
    if args.resume_checkpoint is not None:
        saved = np.load(args.resume_checkpoint, allow_pickle=False)
        initial_model = {name: saved[name] for name in ("w1", "b1", "w2", "b2")}
        saved_samples = int(saved["samples_per_period"])
        if saved_samples != args.samples_per_period:
            parser.error(
                f"checkpoint uses {saved_samples} samples/period, requested "
                f"{args.samples_per_period}"
            )
        if int(saved["spring_count"]) != len(base_k):
            parser.error("checkpoint spring count does not match the active topology")
    phase_count = args.mechanics_refreshes + 1
    phase_iterations = [
        args.iterations // phase_count + (1 if i < args.iterations % phase_count else 0)
        for i in range(phase_count)
    ]
    model, history, scales = initial_model, [], None
    active_train = train
    completed = 0
    for phase_index, count in enumerate(phase_iterations):
        model, phase_history, scales = train_period_model(
            active_train, base_k, args.hidden_dim, count, args.learning_rate,
            args.min_stiffness, args.periods, args.seed + phase_index, args.device,
            args.progress_interval, model, args.observation_variants,
            args.observation_stiffness_log_std, args.training_mode,
            args.training_periods, args.initial_stiffness_log_std,
            stiffness_lower, stiffness_upper, args.stiffness_order_weight,
            args.objective,
        )
        history.extend((completed + i, rmse, loss) for i, rmse, loss in phase_history)
        completed += count
        if phase_index < args.mechanics_refreshes:
            print(f"Refreshing exact mechanics ({phase_index + 1}/{args.mechanics_refreshes})...")
            active_train = refresh_period_basis(
                model, active_train, topology, base_k, args.min_stiffness, scales,
                args.relaxation_steps,
                args.training_periods if args.training_mode == "closed_loop" else 2,
                args.mechanics_batch_size,
                args.mechanics_progress_interval,
                args.initial_stiffness_log_std if args.training_mode == "closed_loop" else 0.0,
                args.seed + phase_index + 1,
                stiffness_lower, stiffness_upper,
            )
    _, stiffness = predict_period_schedule(
        model, test, base_k, args.min_stiffness, args.periods, args.device, scales,
        stiffness_lower=stiffness_lower, stiffness_upper=stiffness_upper,
    )
    torque, force_residual = exact_period_torque(
        test, topology, stiffness, args.relaxation_steps,
        args.mechanics_batch_size, args.mechanics_progress_interval,
    )
    first_rmse = np.sqrt(np.mean((torque[:, 0] - test["target"]) ** 2))
    adapted_rmse = np.sqrt(np.mean((torque[:, 1:] - test["target"][:, None]) ** 2))
    print(f"Held-out first-period default RMSE: {first_rmse:.3f} N*m")
    print(f"Held-out adapted next-period RMSE: {adapted_rmse:.3f} N*m")
    print(f"Mean held-out equilibrium force residual: {np.mean(force_residual):.6f} N")
    output = PROJECT_ROOT / "models" / "period_adaptive_3d" / f"{args.output_name}.npz"
    save_model(output, model, args.output_name, args.min_stiffness, np.inf,
               controller_type="causal_period_adaptive_3d",
               feature_type="previous_period_motion_target_spring_motor",
               channels=np.array(["theta", "theta_dot", "theta_ddot", "target_torque",
                                  "spring_torque", "motor_torque"]),
               first_period_policy="topology_initial_stiffness_no_network_input",
               update_policy="once_at_period_boundary_hold_for_full_period",
               training_alignment="completed_period_input_to_same_period_fit_loss",
               deployment_alignment="completed_period_input_applied_to_following_period",
               samples_per_period=args.samples_per_period, period_seconds=args.period_seconds,
               periods=args.periods, motion_scales=scales, torque_scale=train["torque_scale"],
               motion_mode=args.motion_mode,
               topology=str(args.topology), spring_count=len(base_k), initial_stiffness=base_k,
               training_profiles=args.training_profiles, test_profiles=args.test_profiles,
               training_iterations=args.iterations, learning_rate=args.learning_rate,
               observation_variants=args.observation_variants,
               observation_stiffness_log_std=args.observation_stiffness_log_std,
               min_stiffness_order_change=args.min_stiffness_order_change,
               max_stiffness_order_change=args.max_stiffness_order_change,
               stiffness_lower_bound=stiffness_lower,
               stiffness_upper_bound=stiffness_upper,
               stiffness_parameterization="clipped_positive_softplus",
               stiffness_order_weight=args.stiffness_order_weight,
               training_objective=args.objective,
               mechanics_refreshes=args.mechanics_refreshes,
               training_mode=args.training_mode, training_periods=args.training_periods,
               initial_stiffness_log_std=args.initial_stiffness_log_std,
               resumed_from=(str(args.resume_checkpoint) if args.resume_checkpoint else ""),
               seed=args.seed)
    plot_dir = PROJECT_ROOT / "plots" / "period_adaptive_3d"
    save_period_figures(
        plot_dir, args.output_name, test, torque, stiffness, history, args.figure_examples
    )
    write_evaluation_summary(
        PROJECT_ROOT / "tables" / "period_adaptive_3d" / f"{args.output_name}_summary.csv",
        first_rmse, adapted_rmse, force_residual,
    )
    print(f"Saved {output}")
    print(f"Saved figures to {plot_dir}")


if __name__ == "__main__":
    main()
