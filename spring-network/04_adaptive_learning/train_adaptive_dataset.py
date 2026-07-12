from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only environments can still train with NumPy.
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import (
    ANGLE_DEGREES,
    forward,
    initialize_model,
    initial_stiffnesses,
    save_model,
    spring_torque_basis,
)
from profile_generator import (
    ANGLE_LIMIT_RAD,
    DEFAULT_TORQUE_LIMIT_NM,
    TERRAIN_FAMILIES,
    generate_profile_parameters,
    generate_terrain_profile_parameters,
    profile_torque,
)
from topology_loader import DEFAULT_TOPOLOGY_PATH, load_network


def integrate_trapezoid(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return np.trapz(y, x)


def smooth_noise(rng, samples, scale):
    raw = rng.normal(0.0, scale, size=samples)
    kernel_size = min(13, samples if samples % 2 == 1 else samples - 1)
    kernel_size = max(kernel_size, 1)
    x = np.linspace(-2.5, 2.5, kernel_size)
    kernel = np.exp(-0.5 * x**2)
    kernel /= np.sum(kernel)
    return np.convolve(raw, kernel, mode="same")


def add_irregular_bumps(rng, t, theta, count, max_height):
    bumped = theta.copy()
    duration = float(t[-1] - t[0])
    for _ in range(count):
        center = rng.uniform(t[0] + 0.1 * duration, t[-1] - 0.1 * duration)
        width = rng.uniform(0.035, 0.14)
        height = rng.uniform(-max_height, max_height)
        bumped += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    return bumped


def generate_motion_trajectory(params, duration, samples, seed):
    """Generate theta(t), derivatives, and a piecewise-linear target torque."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, samples)
    amp = np.deg2rad(params["amplitude_deg"])
    freq = params["frequency_hz"]
    phase = params["phase"]

    base = amp * np.sin(2.0 * np.pi * freq * t + phase)
    harmonic = 0.18 * amp * np.sin(2.0 * np.pi * 0.5 * freq * t + 0.4 * phase)
    theta = base + params["harmonic_fraction"] * harmonic
    theta = add_irregular_bumps(
        rng,
        t,
        theta,
        params["bump_count"],
        max_height=0.18 * amp,
    )
    theta += smooth_noise(rng, samples, params["noise_scale"])

    theta = np.clip(theta, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD)
    theta_dot = np.gradient(theta, t)
    theta_ddot = np.gradient(theta_dot, t)
    tau_target = profile_torque(theta, params)
    return t, theta, theta_dot, theta_ddot, tau_target


def motion_window_features(theta, theta_dot, theta_ddot, window_size, scales):
    """Build causal windows from recent motion only, with edge padding at the start."""
    padded = np.column_stack(
        [
            theta / scales["theta"],
            theta_dot / scales["theta_dot"],
            theta_ddot / scales["theta_ddot"],
        ]
    )
    rows = []
    for index in range(len(theta)):
        start = max(0, index - window_size + 1)
        window = padded[start : index + 1]
        if len(window) < window_size:
            pad = np.repeat(window[:1], window_size - len(window), axis=0)
            window = np.vstack([pad, window])
        rows.append(window.reshape(-1))
    return np.asarray(rows, dtype=float)


def profile_parameter_features(params, samples):
    """Repeat normalized torque-profile knot parameters for every trajectory sample."""
    theta_features = np.asarray(params["knots_theta"], dtype=float) / ANGLE_LIMIT_RAD
    tau_features = np.asarray(params["knots_tau"], dtype=float) / DEFAULT_TORQUE_LIMIT_NM
    features = np.concatenate((theta_features, tau_features))
    return np.tile(features, (samples, 1))


def trajectory_features(theta, theta_dot, theta_ddot, params, window_size, scales, include_profile=True):
    motion_features = motion_window_features(theta, theta_dot, theta_ddot, window_size, scales)
    if not include_profile:
        return motion_features
    profile_features = profile_parameter_features(params, len(theta))
    return np.hstack((motion_features, profile_features))


def print_progress(label, current, total, interval):
    if interval <= 0:
        return
    if current == 1 or current == total or current % interval == 0:
        print(f"{label}: {current}/{total}")


def select_training_device(requested):
    if requested == "cpu":
        return "cpu"
    if torch is None:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested, but PyTorch is not installed.")
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_label(device):
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return device


def select_mechanics_backend(requested):
    if requested == "scipy":
        return "scipy"
    if requested == "torch":
        if torch is None:
            raise RuntimeError("Torch mechanics were requested, but PyTorch is not installed.")
        return "torch"
    return "torch" if torch is not None else "scipy"


def interpolate_basis(basis_by_angle, angles_rad, theta):
    basis = np.empty((len(theta), basis_by_angle.shape[1]), dtype=float)
    for spring_index in range(basis_by_angle.shape[1]):
        basis[:, spring_index] = np.interp(
            theta,
            angles_rad,
            basis_by_angle[:, spring_index],
            left=basis_by_angle[0, spring_index],
            right=basis_by_angle[-1, spring_index],
        )
    return basis


def normalization_scales(profile_params, duration, samples, seed, window_size, progress_interval=100):
    theta_values = []
    theta_dot_values = []
    theta_ddot_values = []
    total = len(profile_params)
    for profile_index, params in enumerate(profile_params):
        _, theta, theta_dot, theta_ddot, _ = generate_motion_trajectory(
            params,
            duration,
            samples,
            seed + profile_index,
        )
        theta_values.append(theta)
        theta_dot_values.append(theta_dot)
        theta_ddot_values.append(theta_ddot)
        print_progress("normalization trajectories", profile_index + 1, total, progress_interval)

    def robust_scale(values, fallback):
        joined = np.concatenate(values)
        scale = float(np.percentile(np.abs(joined), 95))
        return max(scale, fallback)

    return {
        "theta": robust_scale(theta_values, np.deg2rad(1.0)),
        "theta_dot": robust_scale(theta_dot_values, 0.1),
        "theta_ddot": robust_scale(theta_ddot_values, 0.5),
        "window_size": int(window_size),
    }


def build_dataset(
    profile_params,
    angles_rad,
    basis_by_angle,
    duration,
    samples,
    window_size,
    scales,
    seed,
    progress_label=None,
    progress_interval=100,
    include_profile=True,
):
    rows = []
    targets = []
    basis_rows = []
    profile_indices = []
    t_rows = []
    theta_rows = []
    theta_dot_rows = []
    theta_ddot_rows = []

    total = len(profile_params)
    for profile_index, params in enumerate(profile_params):
        t, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params,
            duration,
            samples,
            seed + profile_index,
        )
        rows.append(
            trajectory_features(
                theta,
                theta_dot,
                theta_ddot,
                params,
                window_size,
                scales,
                include_profile=include_profile,
            )
        )
        targets.append(tau_target)
        basis_rows.append(interpolate_basis(basis_by_angle, angles_rad, theta))
        profile_indices.append(np.full(samples, profile_index, dtype=int))
        t_rows.append(t)
        theta_rows.append(theta)
        theta_dot_rows.append(theta_dot)
        theta_ddot_rows.append(theta_ddot)
        if progress_label:
            print_progress(progress_label, profile_index + 1, total, progress_interval)

    return {
        "features": np.vstack(rows),
        "target": np.concatenate(targets),
        "basis": np.vstack(basis_rows),
        "profile_indices": np.concatenate(profile_indices),
        "t": np.concatenate(t_rows),
        "theta": np.concatenate(theta_rows),
        "theta_dot": np.concatenate(theta_dot_rows),
        "theta_ddot": np.concatenate(theta_ddot_rows),
        "samples_per_profile": int(samples),
    }


def train_model(
    dataset,
    initial_k,
    hidden_dim,
    iterations,
    learning_rate,
    min_k,
    max_k,
    stiffness_weight,
    seed,
    progress_interval,
    device="auto",
    energy_weight=0.0,
):
    selected_device = select_training_device(device)
    if torch is not None:
        return train_model_torch(
            dataset,
            initial_k,
            hidden_dim,
            iterations,
            learning_rate,
            min_k,
            max_k,
            stiffness_weight,
            seed,
            progress_interval,
            selected_device,
            energy_weight,
        )
    if energy_weight:
        print("Warning: energy_weight is only applied by the PyTorch training path.")
    return train_model_numpy(
        dataset,
        initial_k,
        hidden_dim,
        iterations,
        learning_rate,
        min_k,
        max_k,
        stiffness_weight,
        seed,
        progress_interval,
    )


def train_model_numpy(
    dataset,
    initial_k,
    hidden_dim,
    iterations,
    learning_rate,
    min_k,
    max_k,
    stiffness_weight,
    seed,
    progress_interval,
):
    print("Training device: cpu (NumPy)")
    rng = np.random.default_rng(seed)
    model = initialize_model(
        rng,
        dataset["features"].shape[1],
        hidden_dim,
        dataset["basis"].shape[1],
        initial_k,
        min_k,
        max_k,
    )

    basis = dataset["basis"]
    target = dataset["target"]
    best_model = {name: value.copy() for name, value in model.items()}
    best_loss = float("inf")
    history = {
        "iteration": [],
        "train_rmse": [],
        "loss": [],
        "mse": [],
        "stiffness_penalty": [],
    }
    adam_m = {name: np.zeros_like(value) for name, value in model.items()}
    adam_v = {name: np.zeros_like(value) for name, value in model.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for iteration in range(1, iterations + 1):
        stiffness, cache = forward(model, dataset["features"], min_k, max_k)
        predicted = np.sum(basis * stiffness, axis=1)
        error = predicted - target
        mse = np.mean(error**2)

        stiffness_delta = (stiffness - initial_k) / np.maximum(initial_k, 1.0)
        stiffness_penalty = stiffness_weight * np.mean(stiffness_delta**2)
        loss = mse + stiffness_penalty
        train_rmse = float(np.sqrt(mse))

        history["iteration"].append(iteration)
        history["train_rmse"].append(train_rmse)
        history["loss"].append(float(loss))
        history["mse"].append(float(mse))
        history["stiffness_penalty"].append(float(stiffness_penalty))

        if loss < best_loss:
            best_loss = loss
            best_model = {name: value.copy() for name, value in model.items()}

        d_pred = 2.0 * error / len(error)
        d_stiffness = basis * d_pred[:, None]
        d_stiffness += stiffness_weight * 2.0 * stiffness_delta / (
            stiffness.size * np.maximum(initial_k, 1.0)
        )
        d_logits = d_stiffness * (max_k - min_k) * cache["sigmoid"] * (1.0 - cache["sigmoid"])

        d_w2 = cache["hidden"].T @ d_logits
        d_b2 = np.sum(d_logits, axis=0)
        d_hidden = d_logits @ model["w2"].T
        d_z1 = d_hidden * (1.0 - cache["hidden"] ** 2)
        d_w1 = cache["features"].T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        gradients = {
            "w1": d_w1,
            "b1": d_b1,
            "w2": d_w2,
            "b2": d_b2,
        }
        for name, gradient in gradients.items():
            adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * gradient
            adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * gradient**2
            m_hat = adam_m[name] / (1.0 - beta1**iteration)
            v_hat = adam_v[name] / (1.0 - beta2**iteration)
            model[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if progress_interval > 0 and (iteration == 1 or iteration % progress_interval == 0 or iteration == iterations):
            print(f"iteration {iteration:5d} | train RMSE {train_rmse:8.4f} N*m | loss {loss:9.4f}")

    return best_model, history


def train_model_torch(
    dataset,
    initial_k,
    hidden_dim,
    iterations,
    learning_rate,
    min_k,
    max_k,
    stiffness_weight,
    seed,
    progress_interval,
    device,
    energy_weight,
):
    print(f"Training device: {device_label(device)}")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)
    initial_model = initialize_model(
        rng,
        dataset["features"].shape[1],
        hidden_dim,
        dataset["basis"].shape[1],
        initial_k,
        min_k,
        max_k,
    )

    torch_device = torch.device(device)
    features = torch.as_tensor(dataset["features"], dtype=torch.float32, device=torch_device)
    basis = torch.as_tensor(dataset["basis"], dtype=torch.float32, device=torch_device)
    target = torch.as_tensor(dataset["target"], dtype=torch.float32, device=torch_device)
    theta_dot = torch.as_tensor(dataset["theta_dot"], dtype=torch.float32, device=torch_device)
    initial_k_tensor = torch.as_tensor(initial_k, dtype=torch.float32, device=torch_device)

    parameters = {
        name: torch.tensor(value, dtype=torch.float32, device=torch_device, requires_grad=True)
        for name, value in initial_model.items()
    }
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    best_model = {name: value.copy() for name, value in initial_model.items()}
    best_loss = float("inf")
    history = {
        "iteration": [],
        "train_rmse": [],
        "loss": [],
        "mse": [],
        "stiffness_penalty": [],
        "energy_penalty": [],
        "mean_offload_surrogate": [],
    }

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.tanh(features @ parameters["w1"] + parameters["b1"])
        logits = hidden @ parameters["w2"] + parameters["b2"]
        sig = torch.sigmoid(torch.clamp(logits, -50.0, 50.0))
        stiffness = min_k + (max_k - min_k) * sig
        predicted = torch.sum(basis * stiffness, dim=1)
        error = predicted - target
        mse = torch.mean(error**2)
        baseline_power = torch.relu(target * theta_dot)
        assisted_power = torch.relu((target - predicted) * theta_dot)
        baseline_power_mean = torch.clamp(torch.mean(baseline_power), min=1e-9)
        energy_ratio = torch.mean(assisted_power) / baseline_power_mean
        energy_penalty = energy_weight * mse.detach() * energy_ratio
        stiffness_delta = (stiffness - initial_k_tensor) / torch.clamp(initial_k_tensor, min=1.0)
        stiffness_penalty = stiffness_weight * torch.mean(stiffness_delta**2)
        loss = mse + stiffness_penalty + energy_penalty
        loss.backward()
        optimizer.step()

        mse_value = float(mse.detach().cpu())
        loss_value = float(loss.detach().cpu())
        penalty_value = float(stiffness_penalty.detach().cpu())
        energy_penalty_value = float(energy_penalty.detach().cpu())
        offload_surrogate = float((1.0 - energy_ratio.detach().cpu()).item() * 100.0)
        train_rmse = float(np.sqrt(mse_value))

        history["iteration"].append(iteration)
        history["train_rmse"].append(train_rmse)
        history["loss"].append(loss_value)
        history["mse"].append(mse_value)
        history["stiffness_penalty"].append(penalty_value)
        history["energy_penalty"].append(energy_penalty_value)
        history["mean_offload_surrogate"].append(offload_surrogate)

        if loss_value < best_loss:
            best_loss = loss_value
            best_model = {
                name: value.detach().cpu().numpy().astype(float).copy()
                for name, value in parameters.items()
            }

        if progress_interval > 0 and (iteration == 1 or iteration % progress_interval == 0 or iteration == iterations):
            print(
                f"iteration {iteration:5d} | train RMSE {train_rmse:8.4f} N*m | "
                f"offload surrogate {offload_surrogate:7.2f}% | loss {loss_value:9.4f}"
            )

    return best_model, history


def predict_dataset(model, dataset, min_k, max_k):
    stiffness, _ = forward(model, dataset["features"], min_k, max_k)
    predicted = np.sum(dataset["basis"] * stiffness, axis=1)
    return predicted, stiffness


def stiffness_schedule_from_model(model, dataset, min_k, max_k):
    stiffness, _ = forward(model, dataset["features"], min_k, max_k)
    return stiffness


def torque_from_stiffness(network, theta, stiffness_schedule, relax_internal=True):
    original_stiffness = np.asarray([spring.stiffness_k for spring in network.springs], dtype=float)
    torques = []
    try:
        for theta_value, stiffness_row in zip(theta, stiffness_schedule):
            for spring, stiffness_value in zip(network.springs, stiffness_row):
                spring.stiffness_k = float(stiffness_value)
            _, _, torque = network.evaluate(float(theta_value), relax_internal=relax_internal)
            torques.append(torque)
    finally:
        for spring, stiffness_value in zip(network.springs, original_stiffness):
            spring.stiffness_k = float(stiffness_value)
    return np.asarray(torques, dtype=float)


def relaxed_torque_from_stiffness(network, theta, stiffness_schedule):
    return torque_from_stiffness(network, theta, stiffness_schedule, relax_internal=True)


def torch_torque_from_dataset(
    dataset,
    topology_path,
    stiffness_schedule,
    relax_internal=True,
    device="auto",
    batch_size=8192,
    relaxation_steps=80,
    relaxation_lr=0.03,
    progress_label=None,
    progress_interval=100,
):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    selected_device = select_training_device(device)
    torch_device = torch.device(selected_device)
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, torch_device)
    theta = np.asarray(dataset["theta"], dtype=float)
    stiffness_schedule = np.asarray(stiffness_schedule, dtype=float)
    predicted = np.empty(len(theta), dtype=float)

    total_batches = int(np.ceil(len(theta) / batch_size))
    for batch_index, start in enumerate(range(0, len(theta), batch_size), start=1):
        stop = min(start + batch_size, len(theta))
        theta_batch = torch.as_tensor(theta[start:stop], dtype=torch.float32, device=torch_device)
        stiffness_batch = torch.as_tensor(stiffness_schedule[start:stop], dtype=torch.float32, device=torch_device)
        torque = torch_torque_batch(
            topology,
            theta_batch,
            stiffness_batch,
            relax_internal=relax_internal,
            relaxation_steps=relaxation_steps,
            relaxation_lr=relaxation_lr,
        )
        predicted[start:stop] = torque.detach().cpu().numpy()
        if progress_label:
            print_progress(progress_label, batch_index, total_batches, max(1, progress_interval))

    return predicted


def torch_topology_data(network, device):
    node_names = list(network.nodes)
    node_index = {name: index for index, name in enumerate(node_names)}
    node_types = [network.nodes[name].type for name in node_names]
    local_positions = np.vstack([network.nodes[name].local_position for name in node_names])
    spring_a = np.asarray([node_index[spring.node_a] for spring in network.springs], dtype=np.int64)
    spring_b = np.asarray([node_index[spring.node_b] for spring in network.springs], dtype=np.int64)
    rest_lengths = np.asarray([spring.rest_length for spring in network.springs], dtype=float)
    internal_indices = np.asarray([index for index, kind in enumerate(node_types) if kind == "internal"], dtype=np.int64)
    limb2_indices = np.asarray([index for index, kind in enumerate(node_types) if kind == "limb2"], dtype=np.int64)
    return {
        "node_types": node_types,
        "local_positions": torch.as_tensor(local_positions, dtype=torch.float32, device=device),
        "spring_a": torch.as_tensor(spring_a, dtype=torch.long, device=device),
        "spring_b": torch.as_tensor(spring_b, dtype=torch.long, device=device),
        "rest_lengths": torch.as_tensor(rest_lengths, dtype=torch.float32, device=device),
        "internal_indices": torch.as_tensor(internal_indices, dtype=torch.long, device=device),
        "limb2_indices": torch.as_tensor(limb2_indices, dtype=torch.long, device=device),
    }


def torch_prescribed_positions(topology, theta):
    local = topology["local_positions"]
    positions = local.unsqueeze(0).repeat(len(theta), 1, 1)
    c = torch.cos(theta)
    s = torch.sin(theta)
    for index, node_type in enumerate(topology["node_types"]):
        if node_type == "limb2":
            x = local[index, 0]
            y = local[index, 1]
            positions[:, index, 0] = c * x - s * y
            positions[:, index, 1] = s * x + c * y
    return positions


def torch_spring_energy(topology, positions, stiffness):
    a = topology["spring_a"]
    b = topology["spring_b"]
    rest = topology["rest_lengths"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    stretch = length - rest.unsqueeze(0)
    return 0.5 * torch.sum(stiffness * stretch**2)


def torch_relax_positions(topology, prescribed_positions, stiffness, relaxation_steps, relaxation_lr):
    internal_indices = topology["internal_indices"]
    if internal_indices.numel() == 0 or not relaxation_steps:
        return prescribed_positions

    internal_positions = prescribed_positions[:, internal_indices, :].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([internal_positions], lr=relaxation_lr)
    for _ in range(relaxation_steps):
        optimizer.zero_grad(set_to_none=True)
        positions = prescribed_positions.clone()
        positions[:, internal_indices, :] = internal_positions
        energy = torch_spring_energy(topology, positions, stiffness)
        energy.backward()
        optimizer.step()

    positions = prescribed_positions.clone()
    positions[:, internal_indices, :] = internal_positions.detach()
    return positions


def torch_torque_batch(topology, theta, stiffness, relax_internal, relaxation_steps, relaxation_lr):
    positions = torch_prescribed_positions(topology, theta)
    if relax_internal:
        positions = torch_relax_positions(topology, positions, stiffness, relaxation_steps, relaxation_lr)

    a = topology["spring_a"]
    b = topology["spring_b"]
    rest = topology["rest_lengths"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    direction = delta / length.unsqueeze(2)
    stretch = length - rest.unsqueeze(0)
    force_on_a = stiffness * stretch
    force_on_a = force_on_a.unsqueeze(2) * direction

    torque = torch.zeros(len(theta), dtype=positions.dtype, device=positions.device)
    limb2_indices = set(int(index) for index in topology["limb2_indices"].detach().cpu().numpy())
    for spring_index in range(len(a)):
        node_a = int(a[spring_index])
        node_b = int(b[spring_index])
        if node_a in limb2_indices:
            r = positions[:, node_a, :]
            force = force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
        if node_b in limb2_indices:
            r = positions[:, node_b, :]
            force = -force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
    return torque


def predict_dataset_relaxed(
    model,
    dataset,
    topology_path,
    min_k,
    max_k,
    progress_label=None,
    progress_interval=100,
    mechanics_backend="scipy",
    device="auto",
    mechanics_batch_size=8192,
    relaxation_steps=80,
):
    stiffness = stiffness_schedule_from_model(model, dataset, min_k, max_k)
    predicted = stiffness_schedule_torque(
        dataset,
        topology_path,
        stiffness,
        relax_internal=True,
        progress_label=progress_label,
        progress_interval=progress_interval,
        mechanics_backend=mechanics_backend,
        device=device,
        mechanics_batch_size=mechanics_batch_size,
        relaxation_steps=relaxation_steps,
    )
    return predicted, stiffness


def predict_dataset_with_mechanics(
    model,
    dataset,
    topology_path,
    min_k,
    max_k,
    relax_internal,
    progress_label=None,
    progress_interval=100,
    mechanics_backend="scipy",
    device="auto",
    mechanics_batch_size=8192,
    relaxation_steps=80,
):
    stiffness = stiffness_schedule_from_model(model, dataset, min_k, max_k)
    return stiffness_schedule_torque(
        dataset,
        topology_path,
        stiffness,
        relax_internal=relax_internal,
        progress_label=progress_label,
        progress_interval=progress_interval,
        mechanics_backend=mechanics_backend,
        device=device,
        mechanics_batch_size=mechanics_batch_size,
        relaxation_steps=relaxation_steps,
    ), stiffness


def stiffness_schedule_torque(
    dataset,
    topology_path,
    stiffness_schedule,
    relax_internal,
    progress_label=None,
    progress_interval=100,
    mechanics_backend="scipy",
    device="auto",
    mechanics_batch_size=8192,
    relaxation_steps=80,
):
    if select_mechanics_backend(mechanics_backend) == "torch":
        return torch_torque_from_dataset(
            dataset,
            topology_path,
            stiffness_schedule,
            relax_internal=relax_internal,
            device=device,
            batch_size=mechanics_batch_size,
            relaxation_steps=relaxation_steps,
            progress_label=progress_label,
            progress_interval=progress_interval,
        )

    predicted = np.empty(len(dataset["theta"]), dtype=float)
    samples = dataset["samples_per_profile"]
    profile_count = len(dataset["theta"]) // samples
    for profile_index in range(profile_count):
        start = profile_index * samples
        stop = start + samples
        network, _ = load_network(topology_path)
        predicted[start:stop] = torque_from_stiffness(
            network,
            dataset["theta"][start:stop],
            stiffness_schedule[start:stop],
            relax_internal=relax_internal,
        )
        if progress_label:
            print_progress(progress_label, profile_index + 1, profile_count, progress_interval)
    return predicted


def fixed_stiffness_torque(
    dataset,
    topology_path,
    stiffness,
    relax_internal,
    progress_label=None,
    progress_interval=100,
    mechanics_backend="scipy",
    device="auto",
    mechanics_batch_size=8192,
    relaxation_steps=80,
):
    schedule = np.tile(stiffness, (len(dataset["theta"]), 1))
    return stiffness_schedule_torque(
        dataset,
        topology_path,
        schedule,
        relax_internal=relax_internal,
        progress_label=progress_label,
        progress_interval=progress_interval,
        mechanics_backend=mechanics_backend,
        device=device,
        mechanics_batch_size=mechanics_batch_size,
        relaxation_steps=relaxation_steps,
    )


def fixed_stiffness_relaxed_torque(
    dataset,
    topology_path,
    stiffness,
    progress_label=None,
    progress_interval=100,
    mechanics_backend="scipy",
    device="auto",
    mechanics_batch_size=8192,
    relaxation_steps=80,
):
    return fixed_stiffness_torque(
        dataset,
        topology_path,
        stiffness,
        relax_internal=True,
        progress_label=progress_label,
        progress_interval=progress_interval,
        mechanics_backend=mechanics_backend,
        device=device,
        mechanics_batch_size=mechanics_batch_size,
        relaxation_steps=relaxation_steps,
    )


def energy_offload(t, theta_dot, target, predicted):
    residual = target - predicted
    baseline_power = np.maximum(0.0, target * theta_dot)
    assisted_power = np.maximum(0.0, residual * theta_dot)
    baseline_energy = float(integrate_trapezoid(baseline_power, t))
    assisted_energy = float(integrate_trapezoid(assisted_power, t))
    if abs(baseline_energy) < 1e-12:
        return 0.0
    return 100.0 * (baseline_energy - assisted_energy) / baseline_energy


def summarize_profiles(profile_params, dataset, predicted):
    rows = []
    samples = dataset["samples_per_profile"]
    for profile_index, params in enumerate(profile_params):
        start = profile_index * samples
        stop = start + samples
        target = dataset["target"][start:stop]
        pred = predicted[start:stop]
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        rows.append(
            {
                "profile": params["name"],
                "family": params["family"],
                "rmse_nm": rmse,
                "offload_pct": energy_offload(
                    dataset["t"][start:stop],
                    dataset["theta_dot"][start:stop],
                    target,
                    pred,
                ),
                "mean_abs_residual_nm": float(np.mean(np.abs(target - pred))),
                "peak_abs_residual_nm": float(np.max(np.abs(target - pred))),
            }
        )
    return rows


def print_summary(title, rows):
    rmse = np.asarray([row["rmse_nm"] for row in rows])
    offload = np.asarray([row["offload_pct"] for row in rows])
    print()
    print(title)
    print("-" * len(title))
    print(f"profiles:        {len(rows)}")
    print(f"mean RMSE:       {np.mean(rmse):.4f} N*m")
    print(f"median RMSE:     {np.median(rmse):.4f} N*m")
    print(f"max RMSE:        {np.max(rmse):.4f} N*m")
    print(f"mean offload:    {np.mean(offload):.2f} %")
    print(f"median offload:  {np.median(offload):.2f} %")


def print_worst_cases(rows, count=8):
    print()
    print("Worst held-out trajectories")
    print("---------------------------")
    print("profile                | family              | rmse_Nm | offload_pct | peak_abs_residual_Nm")
    for row in sorted(rows, key=lambda item: item["rmse_nm"], reverse=True)[:count]:
        print(
            f"{row['profile']:22s} | {row['family']:19s} | {row['rmse_nm']:7.3f} | "
            f"{row['offload_pct']:11.3f} | {row['peak_abs_residual_nm']:20.3f}"
        )


def aggregate_profile_rows(model_name, mechanics, rows):
    rmse = np.asarray([row["rmse_nm"] for row in rows], dtype=float)
    offload = np.asarray([row["offload_pct"] for row in rows], dtype=float)
    mean_abs = np.asarray([row["mean_abs_residual_nm"] for row in rows], dtype=float)
    peak_abs = np.asarray([row["peak_abs_residual_nm"] for row in rows], dtype=float)
    return {
        "model": model_name,
        "mechanics": mechanics,
        "profiles": len(rows),
        "mean_rmse_nm": float(np.mean(rmse)),
        "median_rmse_nm": float(np.median(rmse)),
        "mean_offload_pct": float(np.mean(offload)),
        "median_offload_pct": float(np.median(offload)),
        "mean_abs_residual_nm": float(np.mean(mean_abs)),
        "max_peak_abs_residual_nm": float(np.max(peak_abs)),
    }


def print_model_comparison(rows):
    print()
    print("Held-out model/mechanics comparison")
    print("-----------------------------------")
    print("model              | mechanics | profiles | mean_rmse_Nm | median_rmse_Nm | mean_offload_pct")
    for row in rows:
        print(
            f"{row['model']:18s} | {row['mechanics']:9s} | {row['profiles']:8d} | "
            f"{row['mean_rmse_nm']:12.3f} | {row['median_rmse_nm']:14.3f} | "
            f"{row['mean_offload_pct']:16.3f}"
        )


def write_model_comparison_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "mechanics",
        "profiles",
        "mean_rmse_nm",
        "median_rmse_nm",
        "mean_offload_pct",
        "median_offload_pct",
        "mean_abs_residual_nm",
        "max_peak_abs_residual_nm",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: f"{row[column]:.6f}" if isinstance(row[column], float) else row[column]
                    for column in columns
                }
            )


def write_profile_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["profile", "family", "rmse_nm", "offload_pct", "mean_abs_residual_nm", "peak_abs_residual_nm"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_torque_trace_rows(path, profile_params, dataset, predicted, stiffness, network):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = dataset["samples_per_profile"]
    spring_columns = [f"k_{spring.node_a}_to_{spring.node_b}" for spring in network.springs]
    columns = [
        "profile",
        "family",
        "sample_index",
        "t",
        "theta",
        "theta_dot",
        "theta_ddot",
        "target_torque_nm",
        "spring_torque_nm",
        "residual_torque_nm",
        *spring_columns,
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for profile_index, params in enumerate(profile_params):
            start = profile_index * samples
            stop = start + samples
            for sample_index, row_index in enumerate(range(start, stop)):
                row = {
                    "profile": params["name"],
                    "family": params["family"],
                    "sample_index": sample_index,
                    "t": f"{dataset['t'][row_index]:.10f}",
                    "theta": f"{dataset['theta'][row_index]:.10f}",
                    "theta_dot": f"{dataset['theta_dot'][row_index]:.10f}",
                    "theta_ddot": f"{dataset['theta_ddot'][row_index]:.10f}",
                    "target_torque_nm": f"{dataset['target'][row_index]:.10f}",
                    "spring_torque_nm": f"{predicted[row_index]:.10f}",
                    "residual_torque_nm": f"{dataset['target'][row_index] - predicted[row_index]:.10f}",
                }
                for column, value in zip(spring_columns, stiffness[row_index]):
                    row[column] = f"{value:.10f}"
                writer.writerow(row)


def plot_test_examples(path, model, test_params, angles_rad, basis_by_angle, duration, samples, window_size, scales, min_k, max_k, seed, topology_path, include_profile=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for profile_index, (ax, params) in enumerate(zip(axes, test_params[:6])):
        network, _ = load_network(topology_path)
        dataset = build_dataset(
            [params],
            angles_rad,
            basis_by_angle,
            duration,
            samples,
            window_size,
            scales,
            seed + profile_index,
            include_profile=include_profile,
        )
        stiffness = stiffness_schedule_from_model(model, dataset, min_k, max_k)
        predicted = relaxed_torque_from_stiffness(network, dataset["theta"], stiffness)
        ax.plot(dataset["t"], dataset["target"], "k--", linewidth=1.8, label="target")
        ax.plot(dataset["t"], predicted, linewidth=1.5, label="learned")
        ax.set_title(f"{params['family']} / {params['name']}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("torque [N*m]")
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.savefig(path, dpi=160)


def plot_training_convergence(path, history, train_baseline_rmse, test_baseline_rmse):
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.asarray(history["iteration"], dtype=float)
    train_rmse = np.asarray(history["train_rmse"], dtype=float)
    loss = np.asarray(history["loss"], dtype=float)
    best_index = int(np.argmin(loss))

    has_offload = "mean_offload_surrogate" in history and history["mean_offload_surrogate"]
    row_count = 3 if has_offload else 2
    fig, axes = plt.subplots(row_count, 1, figsize=(9, 9 if has_offload else 7), sharex=True, constrained_layout=True)
    axes[0].plot(iterations, train_rmse, linewidth=1.8, label="train RMSE")
    axes[0].axhline(train_baseline_rmse, color="0.35", linestyle="--", linewidth=1.2, label="fixed baseline train RMSE")
    axes[0].axhline(test_baseline_rmse, color="0.55", linestyle=":", linewidth=1.2, label="fixed baseline test RMSE")
    axes[0].scatter(iterations[best_index], train_rmse[best_index], color="tab:red", s=28, zorder=3, label="best loss")
    axes[0].set_ylabel("RMSE [N*m]")
    axes[0].set_title("Adaptive training convergence")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(iterations, loss, linewidth=1.8, label="total loss")
    axes[1].plot(iterations, history["mse"], linewidth=1.2, label="MSE")
    axes[1].plot(iterations, history["stiffness_penalty"], linewidth=1.2, label="stiffness penalty")
    axes[1].scatter(iterations[best_index], loss[best_index], color="tab:red", s=28, zorder=3)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    if has_offload:
        axes[2].plot(iterations, history["mean_offload_surrogate"], linewidth=1.8, color="tab:green")
        axes[2].axhline(0.0, color="0.65", linewidth=1.0)
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("offload surrogate [%]")
        axes[2].grid(True, alpha=0.25)

    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Train adaptive spring stiffnesses on generated piecewise-linear torque profiles.")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_PATH, help="Starting topology JSON file.")
    parser.add_argument(
        "--profile-set",
        choices=["terrain", "arbitrary"],
        default="terrain",
        help="Use separated terrain-family piecewise profiles or arbitrary random piecewise profiles.",
    )
    parser.add_argument("--profiles-per-family", type=int, default=4000, help="Training profiles per terrain family.")
    parser.add_argument("--test-profiles-per-family", type=int, default=400, help="Held-out test profiles per terrain family.")
    parser.add_argument("--train-profiles", type=int, default=12000, help="Training trajectories for --profile-set arbitrary.")
    parser.add_argument("--test-profiles", type=int, default=1200, help="Held-out trajectories for --profile-set arbitrary.")
    parser.add_argument("--duration", type=float, default=5.0, help="Trajectory duration in seconds.")
    parser.add_argument("--samples", type=int, default=160, help="Samples per generated trajectory.")
    parser.add_argument("--window-size", type=int, default=10, help="Recent motion samples used as neural-network input.")
    parser.add_argument(
        "--input-mode",
        choices=["profile", "motion"],
        default="profile",
        help="Use motion plus target knots, or causal motion history only.",
    )
    parser.add_argument("--output-name", default="adaptive_trained_model", help="Stem used for model, table, and plot outputs.")
    parser.add_argument("--iterations", type=int, default=5000, help="Gradient descent iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam step size.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden units in the neural stiffness model.")
    parser.add_argument("--min-stiffness", type=float, default=1.0, help="Minimum learned stiffness in N/m.")
    parser.add_argument("--max-stiffness", type=float, default=800.0, help="Maximum learned stiffness in N/m.")
    parser.add_argument("--stiffness-weight", type=float, default=2e-4, help="Penalty for moving far from baseline stiffnesses.")
    parser.add_argument("--energy-weight", type=float, default=0.35, help="Weight for reducing positive residual motor power during training.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N profiles or optimizer iterations.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device for the neural optimizer. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--mechanics-backend",
        choices=["auto", "torch", "scipy"],
        default="auto",
        help="Mechanics backend for baseline/adaptive evaluation. 'auto' uses torch when available.",
    )
    parser.add_argument("--mechanics-batch-size", type=int, default=8192, help="Samples per GPU mechanics batch.")
    parser.add_argument("--relaxation-steps", type=int, default=80, help="PyTorch internal-node relaxation steps per mechanics batch.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    args = parser.parse_args()
    mechanics_backend = select_mechanics_backend(args.mechanics_backend)
    include_profile = args.input_mode == "profile"

    network, topology = load_network(args.topology)
    angles_rad = np.radians(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles_rad, relax_internal=True)
    base_k = initial_stiffnesses(network)

    rng = np.random.default_rng(args.seed)
    if args.profile_set == "terrain":
        train_params = generate_terrain_profile_parameters(rng, args.profiles_per_family, TERRAIN_FAMILIES)
        test_params = generate_terrain_profile_parameters(rng, args.test_profiles_per_family, TERRAIN_FAMILIES)
    else:
        all_params = generate_profile_parameters(rng, args.train_profiles + args.test_profiles)
        train_params = all_params[: args.train_profiles]
        test_params = all_params[args.train_profiles :]

    scales = normalization_scales(
        train_params,
        args.duration,
        args.samples,
        args.seed + 10_000,
        args.window_size,
        progress_interval=args.progress_interval,
    )
    train_dataset = build_dataset(
        train_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.seed + 20_000,
        progress_label="training dataset profiles",
        progress_interval=args.progress_interval,
        include_profile=include_profile,
    )
    test_dataset = build_dataset(
        test_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.seed + 30_000,
        progress_label="test dataset profiles",
        progress_interval=args.progress_interval,
        include_profile=include_profile,
    )

    baseline_fixed_train = fixed_stiffness_relaxed_torque(
        train_dataset,
        args.topology,
        base_k,
        progress_label="fixed baseline on train set",
        progress_interval=args.progress_interval,
        mechanics_backend=mechanics_backend,
        device=args.device,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
    )
    baseline_fixed_test = fixed_stiffness_relaxed_torque(
        test_dataset,
        args.topology,
        base_k,
        progress_label="fixed baseline on test set",
        progress_interval=args.progress_interval,
        mechanics_backend=mechanics_backend,
        device=args.device,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
    )
    train_baseline_rmse = float(np.sqrt(np.mean((baseline_fixed_train - train_dataset["target"]) ** 2)))
    test_baseline_rmse = float(np.sqrt(np.mean((baseline_fixed_test - test_dataset["target"]) ** 2)))

    print(f"Loaded topology: {topology['name']}")
    print(f"Profile set: {args.profile_set}")
    if args.profile_set == "terrain":
        print(f"Terrain families: {', '.join(TERRAIN_FAMILIES)}")
        print(f"Profiles per terrain family: train {args.profiles_per_family} | test {args.test_profiles_per_family}")
    print(f"Training trajectories: {len(train_params)} | test trajectories: {len(test_params)}")
    print(f"Samples per trajectory: {args.samples} | motion window: {args.window_size} samples")
    feature_description = f"{args.window_size} * theta/theta_dot/theta_ddot"
    if include_profile:
        feature_description += " + 5 knot angles + 5 knot torques"
    print(f"Input mode: {args.input_mode}")
    print(f"Feature count: {train_dataset['features'].shape[1]} ({feature_description})")
    print("Training torque basis: relaxed internal-node geometry")
    print(f"Reported metrics: full relaxed network evaluation via {mechanics_backend}")
    print(f"Fixed-stiffness baseline train RMSE: {train_baseline_rmse:.4f} N*m")
    print(f"Fixed-stiffness baseline test RMSE:  {test_baseline_rmse:.4f} N*m")

    model, training_history = train_model(
        dataset=train_dataset,
        initial_k=base_k,
        hidden_dim=args.hidden_dim,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        min_k=args.min_stiffness,
        max_k=args.max_stiffness,
        stiffness_weight=args.stiffness_weight,
        seed=args.seed,
        progress_interval=args.progress_interval,
        device=args.device,
        energy_weight=args.energy_weight,
    )

    train_pred, train_stiffness = predict_dataset_relaxed(
        model,
        train_dataset,
        args.topology,
        args.min_stiffness,
        args.max_stiffness,
        progress_label="adaptive relaxed train evaluation",
        progress_interval=args.progress_interval,
        mechanics_backend=mechanics_backend,
        device=args.device,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
    )
    test_pred, test_stiffness = predict_dataset_relaxed(
        model,
        test_dataset,
        args.topology,
        args.min_stiffness,
        args.max_stiffness,
        progress_label="adaptive relaxed test evaluation",
        progress_interval=args.progress_interval,
        mechanics_backend=mechanics_backend,
        device=args.device,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
    )
    baseline_unrelaxed_test = fixed_stiffness_torque(
        test_dataset,
        args.topology,
        base_k,
        relax_internal=False,
        progress_label="fixed baseline unrelaxed test evaluation",
        progress_interval=args.progress_interval,
        mechanics_backend=mechanics_backend,
        device=args.device,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
    )
    adaptive_unrelaxed_test, _ = predict_dataset_with_mechanics(
        model,
        test_dataset,
        args.topology,
        args.min_stiffness,
        args.max_stiffness,
        relax_internal=False,
        progress_label="adaptive unrelaxed test evaluation",
        progress_interval=args.progress_interval,
        mechanics_backend=mechanics_backend,
        device=args.device,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
    )
    train_rows = summarize_profiles(train_params, train_dataset, train_pred)
    test_rows = summarize_profiles(test_params, test_dataset, test_pred)
    baseline_relaxed_test_rows = summarize_profiles(test_params, test_dataset, baseline_fixed_test)
    baseline_unrelaxed_test_rows = summarize_profiles(test_params, test_dataset, baseline_unrelaxed_test)
    adaptive_unrelaxed_test_rows = summarize_profiles(test_params, test_dataset, adaptive_unrelaxed_test)
    model_comparison_rows = [
        aggregate_profile_rows("fixed_baseline", "relaxed", baseline_relaxed_test_rows),
        aggregate_profile_rows("fixed_baseline", "unrelaxed", baseline_unrelaxed_test_rows),
        aggregate_profile_rows("adaptive", "relaxed", test_rows),
        aggregate_profile_rows("adaptive", "unrelaxed", adaptive_unrelaxed_test_rows),
    ]

    print_summary("Training-set performance", train_rows)
    print_summary("Held-out test performance", test_rows)
    print_worst_cases(test_rows)
    print_model_comparison(model_comparison_rows)

    output_dir = PROJECT_ROOT
    output_name = args.output_name
    model_path = output_dir / "models" / f"{output_name}.npz"
    train_table_path = output_dir / "tables" / f"{output_name}_train_results.csv"
    test_table_path = output_dir / "tables" / f"{output_name}_test_results.csv"
    torque_trace_path = output_dir / "tables" / f"{output_name}_test_torque_trace.csv"
    model_comparison_path = output_dir / "tables" / f"{output_name}_mechanics_comparison.csv"
    figure_path = output_dir / "plots" / "dataset_examples" / f"{output_name}_test_examples.png"
    convergence_path = output_dir / "plots" / "dataset_examples" / f"{output_name}_training_convergence.png"

    save_model(
        model_path,
        model,
        output_name,
        args.min_stiffness,
        args.max_stiffness,
        feature_type="motion_window_profile" if include_profile else "motion_window",
        window_size=args.window_size,
        theta_scale=scales["theta"],
        theta_dot_scale=scales["theta_dot"],
        theta_ddot_scale=scales["theta_ddot"],
        profile_angle_scale=ANGLE_LIMIT_RAD,
        profile_torque_scale=DEFAULT_TORQUE_LIMIT_NM,
        training_device=select_training_device(args.device),
        hidden_dim=args.hidden_dim,
        energy_weight=args.energy_weight,
        profile_set=args.profile_set,
        profiles_per_family=args.profiles_per_family if args.profile_set == "terrain" else 0,
        test_profiles_per_family=args.test_profiles_per_family if args.profile_set == "terrain" else 0,
        mechanics_backend=mechanics_backend,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
        duration=args.duration,
        samples=args.samples,
    )
    write_profile_rows(train_table_path, train_rows)
    write_profile_rows(test_table_path, test_rows)
    write_torque_trace_rows(torque_trace_path, test_params, test_dataset, test_pred, test_stiffness, network)
    write_model_comparison_rows(model_comparison_path, model_comparison_rows)
    plot_test_examples(
        figure_path,
        model,
        test_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.min_stiffness,
        args.max_stiffness,
        args.seed + 40_000,
        args.topology,
        include_profile=include_profile,
    )
    plot_training_convergence(
        convergence_path,
        training_history,
        train_baseline_rmse,
        test_baseline_rmse,
    )

    print()
    print(f"Saved dataset-trained model to {model_path}")
    print(f"Saved train results to {train_table_path}")
    print(f"Saved test results to {test_table_path}")
    print(f"Saved test torque trace to {torque_trace_path}")
    print(f"Saved mechanics comparison to {model_comparison_path}")
    print(f"Saved test example plot to {figure_path}")
    print(f"Saved training convergence plot to {convergence_path}")
    plt.show()


if __name__ == "__main__":
    main()
