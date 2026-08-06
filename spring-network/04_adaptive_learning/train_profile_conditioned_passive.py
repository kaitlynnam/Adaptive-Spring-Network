"""Train one passive spring-stiffness vector per supplied torque-angle profile.

The MLP is evaluated once per profile from its five angle/torque knots.  Its
output is held constant while every sample in that profile is evaluated, so
the learned mechanism is reconfigurable between profiles but passive during
execution.
"""

from pathlib import Path
import argparse
import csv
import sys

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import (  # noqa: E402
    ANGLE_DEGREES,
    forward,
    initial_stiffnesses,
    initialize_model,
    save_model,
    spring_torque_basis,
)
from energy_accounting import (  # noqa: E402
    DEFAULT_MOTORING_EFFICIENCY,
    DEFAULT_REGEN_EFFICIENCY,
    numpy_power_accounting,
    torch_energy_burden_power,
    validate_efficiencies,
)
from profile_generator import (  # noqa: E402
    DEFAULT_TORQUE_LIMIT_NM,
    PROFILE_CLASSIFICATION,
    PROFILE_FAMILIES,
    generate_classified_profile_parameters,
    profile_descriptor,
)
from topology_loader import load_network  # noqa: E402
from passive_mechanics import (  # noqa: E402
    PASSIVE_NETWORK_PRESETS,
    differentiable_relaxed_stiffness_torque,
    generate_motion_trajectory,
    interpolate_basis,
    select_training_device,
    torch_topology_data,
    torch_torque_components_batch,
    torch_torque_from_dataset,
)

TRAINING_NETWORK_PRESETS = PASSIVE_NETWORK_PRESETS


def build_profile_dataset(
    profiles,
    angles_rad,
    basis_by_angle,
    duration,
    samples,
    seed,
    motion_mode="randomized",
    fixed_frequency_hz=None,
    torque_scale=DEFAULT_TORQUE_LIMIT_NM,
):
    """Return profile-level inputs and sample-level mechanics arrays."""
    descriptors = []
    theta_rows = []
    theta_dot_rows = []
    target_rows = []
    basis_rows = []
    time_rows = []
    for index, profile in enumerate(profiles):
        t, theta, theta_dot, _, target = generate_motion_trajectory(
            profile,
            duration,
            samples,
            seed + index,
            motion_mode=motion_mode,
            fixed_frequency_hz=fixed_frequency_hz,
        )
        descriptors.append(profile_descriptor(profile, torque_scale))
        theta_rows.append(theta)
        theta_dot_rows.append(theta_dot)
        target_rows.append(target)
        basis_rows.append(interpolate_basis(basis_by_angle, angles_rad, theta))
        time_rows.append(t)
    return {
        "profile_features": np.asarray(descriptors, dtype=float),
        "theta": np.asarray(theta_rows, dtype=float),
        "theta_dot": np.asarray(theta_dot_rows, dtype=float),
        "target": np.asarray(target_rows, dtype=float),
        "basis": np.asarray(basis_rows, dtype=float),
        "t": np.asarray(time_rows, dtype=float),
        "samples_per_profile": int(samples),
        "torque_scale": float(torque_scale),
    }


def expand_profile_stiffness(stiffness, samples):
    """Broadcast one stiffness vector across a profile without changing it."""
    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.ndim != 2:
        raise ValueError("profile stiffness must have shape [profiles, springs]")
    return np.broadcast_to(stiffness[:, None, :], (len(stiffness), samples, stiffness.shape[1]))


def predict_profile_stiffness(
    model, dataset, min_k, max_k, unbounded_stiffness=False
):
    if not unbounded_stiffness:
        stiffness, _ = forward(model, dataset["profile_features"], min_k, max_k)
        return stiffness
    features = dataset["profile_features"]
    hidden = np.tanh(features @ model["w1"] + model["b1"])
    logits = hidden @ model["w2"] + model["b2"]
    softplus = np.maximum(logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    return min_k + softplus


def initialize_unbounded_model(rng, input_dim, hidden_dim, output_dim, initial_k, min_k):
    shifted = np.maximum(np.asarray(initial_k) - min_k, 1e-6)
    bias = shifted.copy()
    small = shifted <= 20.0
    bias[small] = np.log(np.expm1(shifted[small]))
    return {
        "w1": rng.normal(0.0, 0.15, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0.0, 0.02, size=(hidden_dim, output_dim)),
        "b2": bias,
    }


def surrogate_torque(profile_stiffness, dataset):
    schedule = expand_profile_stiffness(
        profile_stiffness, dataset["samples_per_profile"]
    )
    return np.sum(dataset["basis"] * schedule, axis=2)


def train_profile_model(
    dataset,
    initial_k,
    hidden_dim,
    iterations,
    learning_rate,
    min_k,
    max_k,
    stiffness_weight,
    energy_weight,
    peak_weight,
    motoring_efficiency,
    regen_efficiency,
    seed,
    device="auto",
    progress_interval=100,
    initial_model=None,
    unbounded_stiffness=False,
):
    """Optimize a deterministic profile-to-passive-stiffness MLP."""
    if torch is None:
        raise RuntimeError("PyTorch is required for profile-conditioned training.")
    selected_device = select_training_device(device)
    torch_device = torch.device(selected_device)
    if initial_model is None:
        rng = np.random.default_rng(seed)
        initializer = initialize_unbounded_model if unbounded_stiffness else initialize_model
        initialize_args = (
            rng, dataset["profile_features"].shape[1], hidden_dim,
            dataset["basis"].shape[2], initial_k, min_k
        )
        model = (
            initializer(*initialize_args)
            if unbounded_stiffness
            else initializer(*initialize_args, max_k)
        )
    else:
        model = {name: np.asarray(value, dtype=float).copy() for name, value in initial_model.items()}
    parameters = {
        name: torch.tensor(value, dtype=torch.float32, device=torch_device, requires_grad=True)
        for name, value in model.items()
    }
    features = torch.as_tensor(dataset["profile_features"], dtype=torch.float32, device=torch_device)
    basis = torch.as_tensor(dataset["basis"], dtype=torch.float32, device=torch_device)
    target = torch.as_tensor(dataset["target"], dtype=torch.float32, device=torch_device)
    theta_dot = torch.as_tensor(dataset["theta_dot"], dtype=torch.float32, device=torch_device)
    base_k = torch.as_tensor(initial_k, dtype=torch.float32, device=torch_device)
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    baseline_power = torch.clamp(
        torch.mean(
            torch_energy_burden_power(
                target * theta_dot, motoring_efficiency, regen_efficiency
            )
        ),
        min=1e-9,
    )
    best_loss = float("inf")
    best_model = None
    history = []
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(features @ parameters["w1"] + parameters["b1"])
        logits = hidden @ parameters["w2"] + parameters["b2"]
        stiffness = (
            min_k + torch.nn.functional.softplus(logits)
            if unbounded_stiffness
            else min_k + (max_k - min_k) * torch.sigmoid(
                torch.clamp(logits, -50.0, 50.0)
            )
        )
        spring_torque = torch.sum(basis * stiffness[:, None, :], dim=2)
        residual = target - spring_torque
        mse = torch.mean(residual**2)
        assisted_power = torch.mean(
            torch_energy_burden_power(
                residual * theta_dot, motoring_efficiency, regen_efficiency
            )
        )
        assisted_fraction = assisted_power / baseline_power
        peak = torch.mean(torch.logsumexp(torch.abs(residual) / 10.0, dim=1) * 10.0)
        stiffness_delta = (stiffness - base_k) / torch.clamp(base_k, min=1.0)
        loss = (
            mse
            + energy_weight * mse.detach() * assisted_fraction
            + peak_weight * peak**2
            + stiffness_weight * torch.mean(stiffness_delta**2)
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        rmse = float(torch.sqrt(mse).detach().cpu())
        offload_pct = 100.0 * (1.0 - float(assisted_fraction.detach().cpu()))
        history.append((iteration, rmse, loss_value, offload_pct))
        if loss_value < best_loss:
            best_loss = loss_value
            best_model = {
                name: value.detach().cpu().numpy().astype(float).copy()
                for name, value in parameters.items()
            }
        if progress_interval > 0 and (
            iteration == 1 or iteration == iterations or iteration % progress_interval == 0
        ):
            print(
                f"iteration {iteration:5d} | residual RMSE {rmse:8.3f} N*m | "
                f"estimated offload {history[-1][3]:7.2f}% | loss {loss_value:10.3f}"
            )
    return best_model, history


def distill_profile_model(
    dataset,
    oracle_stiffness,
    initial_k,
    hidden_dim,
    iterations,
    learning_rate,
    min_k,
    max_k,
    seed,
    device="auto",
    progress_interval=100,
):
    """Supervise the profile encoder with direct per-profile oracle vectors."""
    if torch is None:
        raise RuntimeError("PyTorch is required for oracle distillation.")
    torch_device = torch.device(select_training_device(device))
    rng = np.random.default_rng(seed)
    model = initialize_model(
        rng,
        dataset["profile_features"].shape[1],
        hidden_dim,
        dataset["basis"].shape[2],
        initial_k,
        min_k,
        max_k,
    )
    parameters = {
        name: torch.tensor(value, dtype=torch.float32, device=torch_device, requires_grad=True)
        for name, value in model.items()
    }
    features = torch.as_tensor(dataset["profile_features"], dtype=torch.float32, device=torch_device)
    labels = torch.as_tensor(oracle_stiffness, dtype=torch.float32, device=torch_device)
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    best_loss = float("inf")
    best_model = None
    stiffness_range = max(max_k - min_k, 1e-9)
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(features @ parameters["w1"] + parameters["b1"])
        logits = hidden @ parameters["w2"] + parameters["b2"]
        predicted = min_k + stiffness_range * torch.sigmoid(torch.clamp(logits, -50.0, 50.0))
        loss = torch.mean(((predicted - labels) / stiffness_range) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_model = {
                name: tensor.detach().cpu().numpy().astype(float).copy()
                for name, tensor in parameters.items()
            }
        if progress_interval > 0 and (
            iteration == 1 or iteration == iterations or iteration % progress_interval == 0
        ):
            mae = float(torch.mean(torch.abs(predicted - labels)).detach().cpu())
            print(f"distill iteration {iteration:5d} | stiffness MAE {mae:8.3f} N/m")
    return best_model


def optimize_profile_stiffness_oracle(
    dataset,
    initial_k,
    iterations,
    learning_rate,
    min_k,
    max_k,
    energy_weight,
    motoring_efficiency,
    regen_efficiency,
    device="auto",
    progress_interval=100,
    unbounded_stiffness=False,
):
    """Directly optimize one constant stiffness vector for every profile.

    This deliberately uses no profile-to-stiffness network.  It measures the
    representational ceiling of the current stiffness basis before exact
    relaxed-mechanics verification.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for oracle optimization.")
    torch_device = torch.device(select_training_device(device))
    basis = torch.as_tensor(dataset["basis"], dtype=torch.float32, device=torch_device)
    target = torch.as_tensor(dataset["target"], dtype=torch.float32, device=torch_device)
    theta_dot = torch.as_tensor(dataset["theta_dot"], dtype=torch.float32, device=torch_device)
    if unbounded_stiffness:
        shifted = np.maximum(np.asarray(initial_k) - min_k, 1e-6)
        initial_logits = shifted.copy()
        small = shifted <= 20.0
        initial_logits[small] = np.log(np.expm1(shifted[small]))
    else:
        scaled = np.clip(
            (np.asarray(initial_k) - min_k) / max(max_k - min_k, 1e-9),
            1e-6,
            1.0 - 1e-6,
        )
        initial_logits = np.log(scaled / (1.0 - scaled))
    if initial_logits.ndim == 1:
        initial_logits = np.repeat(initial_logits[None, :], len(target), axis=0)
    elif initial_logits.shape != (len(target), basis.shape[2]):
        raise ValueError("initial_k must be [springs] or [profiles, springs]")
    logits = torch.tensor(
        initial_logits,
        dtype=torch.float32,
        device=torch_device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    baseline_power = torch.clamp(
        torch.mean(
            torch_energy_burden_power(
                target * theta_dot, motoring_efficiency, regen_efficiency
            ),
            dim=1,
        ),
        min=1e-9,
    )
    best_loss = float("inf")
    best_stiffness = None
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        stiffness = (
            min_k + torch.nn.functional.softplus(logits)
            if unbounded_stiffness
            else min_k + (max_k - min_k) * torch.sigmoid(
                torch.clamp(logits, -50.0, 50.0)
            )
        )
        spring_torque = torch.sum(basis * stiffness[:, None, :], dim=2)
        residual = target - spring_torque
        profile_mse = torch.mean(residual**2, dim=1)
        assisted_power = torch.mean(
            torch_energy_burden_power(
                residual * theta_dot, motoring_efficiency, regen_efficiency
            ),
            dim=1,
        )
        assisted_fraction = assisted_power / baseline_power
        loss = torch.mean(profile_mse + energy_weight * profile_mse.detach() * assisted_fraction)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_stiffness = stiffness.detach().cpu().numpy().astype(float).copy()
        if progress_interval > 0 and (
            iteration == 1 or iteration == iterations or iteration % progress_interval == 0
        ):
            print(
                f"oracle iteration {iteration:5d} | residual RMSE "
                f"{float(torch.sqrt(torch.mean(profile_mse)).detach().cpu()):8.3f} N*m | "
                f"mean estimated offload "
                f"{100.0 * (1.0 - float(torch.mean(assisted_fraction).detach().cpu())):7.2f}%"
            )
    return best_stiffness


def exact_mechanics_finetune(
    model,
    dataset,
    topology_path,
    min_k,
    max_k,
    updates,
    learning_rate,
    profile_batch_size,
    samples_per_profile,
    relaxation_steps,
    relaxation_step_size,
    relaxation_max_step,
    motoring_efficiency,
    regen_efficiency,
    seed,
    device="auto",
):
    """Energy-first minibatch fine-tuning through unrolled relaxed mechanics."""
    if torch is None:
        raise RuntimeError("PyTorch is required for exact-mechanics fine-tuning.")
    torch_device = torch.device(select_training_device(device))
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, torch_device)
    parameters = {
        name: torch.tensor(value, dtype=torch.float32, device=torch_device, requires_grad=True)
        for name, value in model.items()
    }
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    features = torch.as_tensor(dataset["profile_features"], dtype=torch.float32, device=torch_device)
    theta_all = torch.as_tensor(dataset["theta"], dtype=torch.float32, device=torch_device)
    target_all = torch.as_tensor(dataset["target"], dtype=torch.float32, device=torch_device)
    velocity_all = torch.as_tensor(dataset["theta_dot"], dtype=torch.float32, device=torch_device)
    rng = np.random.default_rng(seed)
    profile_count, total_samples = dataset["target"].shape
    for update in range(1, updates + 1):
        profile_indices = rng.choice(
            profile_count, size=min(profile_batch_size, profile_count), replace=False
        )
        sample_indices = np.sort(
            rng.choice(
                total_samples,
                size=min(samples_per_profile, total_samples),
                replace=False,
            )
        )
        p = torch.as_tensor(profile_indices, dtype=torch.long, device=torch_device)
        s = torch.as_tensor(sample_indices, dtype=torch.long, device=torch_device)
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(features[p] @ parameters["w1"] + parameters["b1"])
        logits = hidden @ parameters["w2"] + parameters["b2"]
        stiffness = min_k + (max_k - min_k) * torch.sigmoid(torch.clamp(logits, -50.0, 50.0))
        theta = theta_all[p][:, s]
        target = target_all[p][:, s]
        velocity = velocity_all[p][:, s]
        expanded_k = stiffness[:, None, :].expand(-1, len(s), -1)
        torque = differentiable_relaxed_stiffness_torque(
            topology,
            theta.reshape(-1),
            expanded_k.reshape(-1, expanded_k.shape[2]),
            relaxation_steps,
            relaxation_step_size,
            relaxation_max_step,
        ).reshape(theta.shape)
        residual = target - torque
        baseline = torch.mean(
            torch_energy_burden_power(
                target * velocity, motoring_efficiency, regen_efficiency
            )
        ).clamp_min(1e-9)
        assisted = torch.mean(
            torch_energy_burden_power(
                residual * velocity, motoring_efficiency, regen_efficiency
            )
        )
        normalized_mse = torch.mean(residual**2) / torch.mean(target**2).clamp_min(1e-9)
        loss = assisted / baseline + 0.1 * normalized_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(), 10.0)
        optimizer.step()
        if update == 1 or update == updates or update % max(updates // 10, 1) == 0:
            print(
                f"exact update {update:4d}/{updates} | minibatch energy offload "
                f"{100.0 * (1.0 - float((assisted / baseline).detach().cpu())):8.3f}%"
            )
    return {
        name: value.detach().cpu().numpy().astype(float).copy()
        for name, value in parameters.items()
    }


def relaxed_profile_torque(
    dataset, topology_path, profile_stiffness, device, batch_size, relaxation_steps
):
    schedule = expand_profile_stiffness(
        profile_stiffness, dataset["samples_per_profile"]
    )
    flat_dataset = {"theta": dataset["theta"].reshape(-1)}
    return torch_torque_from_dataset(
        flat_dataset,
        topology_path,
        schedule.reshape(-1, schedule.shape[2]),
        relax_internal=True,
        device=device,
        batch_size=batch_size,
        relaxation_steps=relaxation_steps,
    ).reshape(dataset["target"].shape)


def refresh_profile_torque_basis(
    dataset,
    topology_path,
    profile_stiffness,
    device="auto",
    batch_size=1024,
    relaxation_steps=300,
):
    """Rebuild a local per-spring basis at constant profile stiffnesses."""
    if torch is None:
        raise RuntimeError("PyTorch is required for basis refresh.")
    torch_device = torch.device(select_training_device(device))
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, torch_device)
    schedule = expand_profile_stiffness(
        profile_stiffness, dataset["samples_per_profile"]
    ).reshape(-1, profile_stiffness.shape[1])
    theta = dataset["theta"].reshape(-1)
    components = np.empty_like(schedule)
    for start in range(0, len(theta), batch_size):
        stop = min(start + batch_size, len(theta))
        theta_batch = torch.as_tensor(
            theta[start:stop], dtype=torch.float32, device=torch_device
        )
        stiffness_batch = torch.as_tensor(
            schedule[start:stop], dtype=torch.float32, device=torch_device
        )
        values = torch_torque_components_batch(
            topology,
            theta_batch,
            stiffness_batch,
            True,
            relaxation_steps,
            0.03,
        )
        components[start:stop] = values.detach().cpu().numpy()
    refreshed = dict(dataset)
    refreshed["basis"] = (
        components / np.maximum(schedule, 1e-6)
    ).reshape(dataset["basis"].shape)
    return refreshed


def profile_motor_work(t, torque, theta_dot, motoring_efficiency=1.0, regen_efficiency=0.0):
    """Integrate ideal absolute motor mechanical power over a profile."""
    accounting = numpy_power_accounting(
        np.asarray(torque) * np.asarray(theta_dot),
        motoring_efficiency,
        regen_efficiency,
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(accounting["energy_burden_power"], t))


# Backward-compatible name retained for archived callers.
profile_energy_burden = profile_motor_work


def summary_rows(
    profiles,
    dataset,
    torque,
    profile_stiffness,
    motoring_efficiency=DEFAULT_MOTORING_EFFICIENCY,
    regen_efficiency=DEFAULT_REGEN_EFFICIENCY,
):
    rows = []
    for index, profile in enumerate(profiles):
        residual = dataset["target"][index] - torque[index]
        baseline_burden = profile_motor_work(
            dataset["t"][index],
            dataset["target"][index],
            dataset["theta_dot"][index],
            motoring_efficiency,
            regen_efficiency,
        )
        assisted_burden = profile_motor_work(
            dataset["t"][index],
            residual,
            dataset["theta_dot"][index],
            motoring_efficiency,
            regen_efficiency,
        )
        offload = (
            100.0 * (baseline_burden - assisted_burden) / baseline_burden
            if baseline_burden > 1e-12
            else 0.0
        )
        rows.append(
            {
                "profile": profile["name"],
                "family": profile["family"],
                "residual_rmse_nm": float(np.sqrt(np.mean(residual**2))),
                "peak_abs_residual_nm": float(np.max(np.abs(residual))),
                "mean_abs_residual_nm": float(np.mean(np.abs(residual))),
                "offload_pct": offload,
                "baseline_motor_work_j": baseline_burden,
                "assisted_motor_work_j": assisted_burden,
                "mean_stiffness_n_per_m": float(np.mean(profile_stiffness[index])),
            }
        )
    return rows


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_stiffness_rows(path, profiles, stiffness):
    rows = []
    for profile, values in zip(profiles, stiffness):
        row = {"profile": profile["name"], "family": profile["family"]}
        row.update(
            {f"spring_{index:02d}": float(value) for index, value in enumerate(values)}
        )
        rows.append(row)
    write_rows(path, rows)


def main():
    parser = argparse.ArgumentParser(
        description="Train a profile-conditioned passive spring network."
    )
    parser.add_argument("--network", choices=sorted(TRAINING_NETWORK_PRESETS), default="fan")
    parser.add_argument("--topology", default=None)
    parser.add_argument("--profiles-per-family", type=int, default=2000)
    parser.add_argument("--test-profiles-per-family", type=int, default=400)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--motion-mode", choices=["randomized", "triangular"], default="randomized")
    parser.add_argument("--fixed-frequency-hz", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--min-stiffness", type=float, default=1.0)
    parser.add_argument("--max-stiffness", type=float, default=800.0)
    parser.add_argument(
        "--unbounded-stiffness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use positive softplus outputs with no upper stiffness cap.",
    )
    parser.add_argument("--stiffness-weight", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.35)
    parser.add_argument("--peak-weight", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--mechanics-batch-size", type=int, default=8192)
    parser.add_argument("--relaxation-steps", type=int, default=80)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-name", default="profile_conditioned_passive")
    parser.add_argument(
        "--oracle-distillation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Directly optimize training-profile stiffness labels and pretrain the MLP on them.",
    )
    parser.add_argument("--oracle-iterations", type=int, default=3000)
    parser.add_argument("--distillation-iterations", type=int, default=3000)
    parser.add_argument("--exact-finetune-updates", type=int, default=0)
    parser.add_argument("--exact-finetune-learning-rate", type=float, default=0.0001)
    parser.add_argument("--exact-finetune-profile-batch", type=int, default=8)
    parser.add_argument("--exact-finetune-samples", type=int, default=24)
    parser.add_argument("--exact-finetune-relaxation-steps", type=int, default=20)
    args = parser.parse_args()
    args.motoring_efficiency = 1.0
    args.regen_efficiency = 0.0
    validate_efficiencies(args.motoring_efficiency, args.regen_efficiency)
    if args.fixed_frequency_hz is not None and args.fixed_frequency_hz <= 0:
        parser.error("--fixed-frequency-hz must be positive")
    preset = TRAINING_NETWORK_PRESETS[args.network]
    topology_path = Path(args.topology) if args.topology else Path(preset["topology"])
    network, topology = load_network(topology_path)
    angles_rad = np.radians(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles_rad, relax_internal=True)
    base_k = initial_stiffnesses(network)
    rng = np.random.default_rng(args.seed)
    train_profiles = generate_classified_profile_parameters(rng, args.profiles_per_family)
    test_profiles = generate_classified_profile_parameters(rng, args.test_profiles_per_family)
    dataset_args = (
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
    )
    train = build_profile_dataset(
        train_profiles, *dataset_args, args.seed + 20_000,
        motion_mode=args.motion_mode, fixed_frequency_hz=args.fixed_frequency_hz
    )
    test = build_profile_dataset(
        test_profiles, *dataset_args, args.seed + 30_000,
        motion_mode=args.motion_mode, fixed_frequency_hz=args.fixed_frequency_hz
    )
    print(f"Loaded topology: {topology['name']} ({len(base_k)} springs)")
    print("Input: complete five-knot torque-angle profile (10 values)")
    print("Output: one stiffness vector per profile, held constant for all samples")
    initial_model = None
    if args.oracle_distillation:
        print("Optimizing one oracle stiffness vector per training profile...")
        oracle_labels = optimize_profile_stiffness_oracle(
            train,
            base_k,
            args.oracle_iterations,
            args.learning_rate,
            args.min_stiffness,
            args.max_stiffness,
            args.energy_weight,
            args.motoring_efficiency,
            args.regen_efficiency,
            device=args.device,
            progress_interval=args.progress_interval,
        )
        print("Distilling oracle stiffness vectors into the profile encoder...")
        initial_model = distill_profile_model(
            train,
            oracle_labels,
            base_k,
            args.hidden_dim,
            args.distillation_iterations,
            args.learning_rate,
            args.min_stiffness,
            args.max_stiffness,
            args.seed,
            device=args.device,
            progress_interval=args.progress_interval,
        )
    model, history = train_profile_model(
        train,
        base_k,
        args.hidden_dim,
        args.iterations,
        args.learning_rate,
        args.min_stiffness,
        args.max_stiffness,
        args.stiffness_weight,
        args.energy_weight,
        args.peak_weight,
        args.motoring_efficiency,
        args.regen_efficiency,
        args.seed,
        device=args.device,
        progress_interval=args.progress_interval,
        initial_model=initial_model,
        unbounded_stiffness=args.unbounded_stiffness,
    )
    if args.exact_finetune_updates:
        print("Fine-tuning through profile-constant unrolled relaxed mechanics...")
        model = exact_mechanics_finetune(
            model,
            train,
            topology_path,
            args.min_stiffness,
            args.max_stiffness,
            args.exact_finetune_updates,
            args.exact_finetune_learning_rate,
            args.exact_finetune_profile_batch,
            args.exact_finetune_samples,
            args.exact_finetune_relaxation_steps,
            0.0001,
            0.005,
            args.motoring_efficiency,
            args.regen_efficiency,
            args.seed + 70_000,
            device=args.device,
        )
    train_k = predict_profile_stiffness(
        model, train, args.min_stiffness, args.max_stiffness, args.unbounded_stiffness
    )
    test_k = predict_profile_stiffness(
        model, test, args.min_stiffness, args.max_stiffness, args.unbounded_stiffness
    )
    test_torque = relaxed_profile_torque(
        test, topology_path, test_k, args.device,
        args.mechanics_batch_size, args.relaxation_steps
    )
    rows = summary_rows(
        test_profiles,
        test,
        test_torque,
        test_k,
        args.motoring_efficiency,
        args.regen_efficiency,
    )
    mean_rmse = float(np.mean([row["residual_rmse_nm"] for row in rows]))
    mean_offload = float(np.mean([row["offload_pct"] for row in rows]))
    aggregate_baseline = float(
        np.sum([row["baseline_motor_work_j"] for row in rows])
    )
    aggregate_assisted = float(
        np.sum([row["assisted_motor_work_j"] for row in rows])
    )
    aggregate_offload = 100.0 * (
        aggregate_baseline - aggregate_assisted
    ) / max(aggregate_baseline, 1e-12)
    print(f"Held-out mean profile RMSE: {mean_rmse:.3f} N*m")
    print(f"Held-out mean profile offload: {mean_offload:.3f}%")
    print(f"Held-out aggregate energy offload: {aggregate_offload:.3f}%")
    print("Within-profile stiffness change: 0 N/m (constant by construction)")
    model_dir = PROJECT_ROOT / "models" / "profile_conditioned_passive"
    table_dir = PROJECT_ROOT / "tables" / "profile_conditioned_passive"
    save_model(
        model_dir / f"{args.output_name}.npz",
        model,
        args.output_name,
        args.min_stiffness,
        args.max_stiffness,
        controller_type="profile_conditioned_passive",
        feature_type="five_knot_torque_angle_profile",
        input_dim=train["profile_features"].shape[1],
        spring_count=len(base_k),
        samples_per_profile=args.samples,
        profile_classification=PROFILE_CLASSIFICATION,
        profile_families=np.asarray(PROFILE_FAMILIES),
        torque_scale=DEFAULT_TORQUE_LIMIT_NM,
        hidden_dim=args.hidden_dim,
        oracle_distillation=args.oracle_distillation,
        exact_finetune_updates=args.exact_finetune_updates,
        stiffness_parameterization=(
            "positive_unbounded_softplus"
            if args.unbounded_stiffness
            else "bounded_sigmoid"
        ),
        seed=args.seed,
    )
    write_rows(table_dir / f"{args.output_name}_test_results.csv", rows)
    write_rows(
        table_dir / f"{args.output_name}_training_history.csv",
        [
            {"iteration": i, "residual_rmse_nm": r, "loss": loss, "offload_pct": offload}
            for i, r, loss, offload in history
        ],
    )
    write_stiffness_rows(
        table_dir / f"{args.output_name}_test_stiffness.csv",
        test_profiles,
        test_k,
    )
    print(f"Saved model: {model_dir / f'{args.output_name}.npz'}")


if __name__ == "__main__":
    main()
