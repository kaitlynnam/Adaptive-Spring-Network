from pathlib import Path
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
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
from energy_accounting import (
    DEFAULT_MOTORING_EFFICIENCY,
    DEFAULT_REGEN_EFFICIENCY,
    numpy_power_accounting,
    torch_energy_burden_power,
    validate_efficiencies,
)
from profile_generator import (
    ANGLE_LIMIT_RAD,
    PROFILE_CLASSIFICATION,
    TERRAIN_FAMILIES,
    generate_classified_profile_parameters,
    generate_profile_parameters,
    profile_descriptor,
    profile_torque,
)

EXPERIMENT_CUBIC_RATIO = 0.0
EXPERIMENT_CUBIC_REFERENCE_EXTENSION = 0.05
from periodicity_classifier import PERIODICITY_CLASSIFICATION, periodicity_score
from topology_loader import DEFAULT_TOPOLOGY_PATH, load_network


TRAINING_NETWORK_PRESETS = {
    "baseline": {
        "topology": PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "baseline_model.json",
        "output_name": "adaptive_trained_baseline_model",
    },
    "fan": {
        "topology": PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "internal_fan_20_spring_model.json",
        "output_name": "adaptive_trained_internal_fan_20spring_model",
    },
}


def integrate_trapezoid(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return np.trapz(y, x)


def causal_derivative(values, t):
    """Differentiate using only the current and previous samples."""
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if values.shape != t.shape:
        raise ValueError("values and t must have the same shape.")

    derivative = np.zeros_like(values)
    if len(values) < 2:
        return derivative

    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("Trajectory time values must be strictly increasing.")
    derivative[1:] = np.diff(values) / dt
    return derivative


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


def generate_motion_trajectory(
    params, duration, samples, seed, motion_mode="randomized", fixed_frequency_hz=None
):
    """Generate theta(t), derivatives, and a piecewise-linear target torque."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, samples)
    freq = params["frequency_hz"] if fixed_frequency_hz is None else float(fixed_frequency_hz)
    if motion_mode == "triangular":
        # Start at the lower limit, reach the upper limit halfway through each
        # cycle, then return to the lower limit. No future/profile information
        # or random motion disturbances are introduced.
        cycle_fraction = np.mod(freq * t, 1.0)
        theta = ANGLE_LIMIT_RAD * (1.0 - 4.0 * np.abs(cycle_fraction - 0.5))
    elif motion_mode == "randomized":
        amp = np.deg2rad(params["amplitude_deg"])
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
    else:
        raise ValueError("motion_mode must be 'randomized' or 'triangular'")

    theta = np.clip(theta, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD)
    theta_dot = causal_derivative(theta, t)
    theta_ddot = causal_derivative(theta_dot, t)
    tau_target = profile_torque(theta, params)
    return t, theta, theta_dot, theta_ddot, tau_target


def generate_periodicity_profiles(rng, count, duration, samples, seed, periodicity_class):
    """Generate candidates and retain one cycle-repeatability third."""
    if count <= 0:
        raise ValueError("count must be positive")
    if periodicity_class not in {"high", "medium", "low"}:
        raise ValueError("periodicity_class must be high, medium, or low")
    candidates = generate_profile_parameters(rng, 3 * count)
    for profile_index, params in enumerate(candidates):
        t, theta, _, _, torque = generate_motion_trajectory(
            params, duration, samples, seed + profile_index
        )
        metrics = periodicity_score(t, theta, torque, params["frequency_hz"])
        params.update(metrics)
        params["classification"] = PERIODICITY_CLASSIFICATION

    candidates.sort(key=lambda profile: profile["periodicity_score"], reverse=True)
    class_index = {"high": 0, "medium": 1, "low": 2}[periodicity_class]
    start = class_index * count
    selected = candidates[start : start + count]
    family = f"{periodicity_class}_periodicity"
    for index, profile in enumerate(selected):
        profile["family"] = family
        profile["name"] = f"{family}_{index:04d}"
    rng.shuffle(selected)
    return selected


def generate_high_periodicity_profiles(rng, count, duration, samples, seed):
    """Backward-compatible wrapper for the highest-repeatability third."""
    return generate_periodicity_profiles(rng, count, duration, samples, seed, "high")


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


def normalization_scales(profile_params, duration, samples, seed, window_size, progress_interval=100, motion_mode="randomized", fixed_frequency_hz=None):
    theta_values = []
    theta_dot_values = []
    theta_ddot_values = []
    torque_values = []
    total = len(profile_params)
    for profile_index, params in enumerate(profile_params):
        _, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params, duration, samples, seed + profile_index,
            motion_mode=motion_mode, fixed_frequency_hz=fixed_frequency_hz
        )
        theta_values.append(theta)
        theta_dot_values.append(theta_dot)
        theta_ddot_values.append(theta_ddot)
        torque_values.append(tau_target)
        print_progress("normalization trajectories", profile_index + 1, total, progress_interval)

    def robust_scale(values, fallback):
        joined = np.concatenate(values)
        scale = float(np.percentile(np.abs(joined), 95))
        return max(scale, fallback)

    return {
        "theta": robust_scale(theta_values, np.deg2rad(1.0)),
        "theta_dot": robust_scale(theta_dot_values, 0.1),
        "theta_ddot": robust_scale(theta_ddot_values, 0.5),
        "torque": robust_scale(torque_values, 1.0),
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
    stiffness_update_mode="timestep",
    progress_label=None,
    progress_interval=100,
    include_profile_descriptor=True,
    motion_mode="randomized",
    fixed_frequency_hz=None,
):
    rows = []
    targets = []
    basis_rows = []
    profile_indices = []
    t_rows = []
    theta_rows = []
    theta_dot_rows = []
    theta_ddot_rows = []
    update_mask_rows = []

    total = len(profile_params)
    for profile_index, params in enumerate(profile_params):
        t, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params, duration, samples, seed + profile_index,
            motion_mode=motion_mode, fixed_frequency_hz=fixed_frequency_hz
        )
        motion_features = motion_window_features(theta, theta_dot, theta_ddot, window_size, scales)
        if include_profile_descriptor:
            descriptor = profile_descriptor(params, scales["torque"])
            rows.append(np.hstack((motion_features, np.repeat(descriptor[None, :], samples, axis=0))))
        else:
            rows.append(motion_features)
        targets.append(tau_target)
        basis_rows.append(interpolate_basis(basis_by_angle, angles_rad, theta))
        profile_indices.append(np.full(samples, profile_index, dtype=int))
        t_rows.append(t)
        theta_rows.append(theta)
        theta_dot_rows.append(theta_dot)
        theta_ddot_rows.append(theta_ddot)
        if stiffness_update_mode == "period":
            cycle = np.floor((t - t[0]) * params["frequency_hz"] + 1e-9).astype(int)
            update_mask = np.concatenate(([True], cycle[1:] != cycle[:-1]))
        elif stiffness_update_mode == "timestep":
            update_mask = np.ones(samples, dtype=bool)
        else:
            raise ValueError("stiffness_update_mode must be 'timestep' or 'period'")
        update_mask_rows.append(update_mask)
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
        "window_size": int(window_size),
        "torque_scale": float(scales["torque"]),
        "stiffness_update_mode": stiffness_update_mode,
        "update_mask": np.concatenate(update_mask_rows),
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
    motoring_efficiency=DEFAULT_MOTORING_EFFICIENCY,
    regen_efficiency=DEFAULT_REGEN_EFFICIENCY,
    stiffness_change_weight=0.0,
    optimizer_name="adam",
    initial_model=None,
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
            motoring_efficiency,
            regen_efficiency,
            stiffness_change_weight,
            optimizer_name,
            initial_model,
        )
    raise RuntimeError(
        "PyTorch is required for causal torque-history training. Install torch in this environment."
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


def torch_causal_torque_rollout(parameters, features, basis, target, dataset, min_k, max_k):
    """Roll forward using only torque values realized before each prediction."""
    samples = dataset["samples_per_profile"]
    profiles = len(dataset["target"]) // samples
    window_size = dataset["window_size"]
    torque_scale = max(dataset["torque_scale"], 1e-9)
    motion = features.reshape(profiles, samples, -1)
    basis = basis.reshape(profiles, samples, -1)
    target = target.reshape(profiles, samples)
    history = torch.zeros((profiles, window_size, 3), dtype=features.dtype, device=features.device)
    predicted_rows = []
    stiffness_rows = []
    update_mask = torch.as_tensor(
        dataset["update_mask"].reshape(profiles, samples), dtype=torch.bool, device=features.device
    )
    held_stiffness = None

    for sample_index in range(samples):
        inputs = torch.cat((motion[:, sample_index, :], history.reshape(profiles, -1)), dim=1)
        hidden = torch.tanh(inputs @ parameters["w1"] + parameters["b1"])
        logits = hidden @ parameters["w2"] + parameters["b2"]
        sig = torch.sigmoid(torch.clamp(logits, -50.0, 50.0))
        candidate_stiffness = min_k + (max_k - min_k) * sig
        if held_stiffness is None:
            stiffness = candidate_stiffness
        else:
            stiffness = torch.where(
                update_mask[:, sample_index].unsqueeze(1), candidate_stiffness, held_stiffness
            )
        held_stiffness = stiffness
        spring_torque = torch.sum(basis[:, sample_index, :] * stiffness, dim=1)
        motor_torque = target[:, sample_index] - spring_torque
        predicted_rows.append(spring_torque)
        stiffness_rows.append(stiffness)

        realized = torch.stack(
            (target[:, sample_index], spring_torque, motor_torque), dim=1
        ) / torque_scale
        history = torch.cat((history[:, 1:, :], realized.detach().unsqueeze(1)), dim=1)

    predicted = torch.stack(predicted_rows, dim=1).reshape(-1)
    stiffness = torch.stack(stiffness_rows, dim=1).reshape(-1, basis.shape[2])
    return predicted, stiffness


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
    motoring_efficiency,
    regen_efficiency,
    stiffness_change_weight,
    optimizer_name,
    initial_model=None,
):
    print(f"Training device: {device_label(device)}")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    if initial_model is None:
        rng = np.random.default_rng(seed)
        initial_model = initialize_model(
            rng,
            dataset["features"].shape[1] + 3 * dataset["window_size"],
            hidden_dim,
            dataset["basis"].shape[1],
            initial_k,
            min_k,
            max_k,
        )
    else:
        initial_model = {name: np.asarray(value, dtype=float).copy() for name, value in initial_model.items()}

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
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    elif optimizer_name == "sgd":
        # No momentum, adaptive moments, or per-parameter learning rates:
        # this is ordinary full-batch gradient descent.
        optimizer = torch.optim.SGD(parameters.values(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
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
        "stiffness_change_penalty": [],
    }

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted, stiffness = torch_causal_torque_rollout(
            parameters, features, basis, target, dataset, min_k, max_k
        )
        error = predicted - target
        mse = torch.mean(error**2)
        baseline_power = torch_energy_burden_power(
            target * theta_dot, motoring_efficiency, regen_efficiency
        )
        assisted_power = torch_energy_burden_power(
            (target - predicted) * theta_dot, motoring_efficiency, regen_efficiency
        )
        baseline_power_mean = torch.clamp(torch.mean(baseline_power), min=1e-9)
        energy_ratio = torch.mean(assisted_power) / baseline_power_mean
        energy_penalty = energy_weight * mse.detach() * energy_ratio
        stiffness_delta = (stiffness - initial_k_tensor) / torch.clamp(initial_k_tensor, min=1.0)
        stiffness_penalty = stiffness_weight * torch.mean(stiffness_delta**2)
        schedule = stiffness.reshape(-1, dataset["samples_per_profile"], stiffness.shape[1])
        normalized_change = (schedule[:, 1:, :] - schedule[:, :-1, :]) / max(max_k - min_k, 1e-9)
        stiffness_change_penalty = stiffness_change_weight * mse.detach() * torch.mean(
            normalized_change**2
        )
        loss = mse + stiffness_penalty + energy_penalty + stiffness_change_penalty
        loss.backward()
        optimizer.step()

        mse_value = float(mse.detach().cpu())
        loss_value = float(loss.detach().cpu())
        penalty_value = float(stiffness_penalty.detach().cpu())
        energy_penalty_value = float(energy_penalty.detach().cpu())
        offload_surrogate = float((1.0 - energy_ratio.detach().cpu()).item() * 100.0)
        change_penalty_value = float(stiffness_change_penalty.detach().cpu())
        train_rmse = float(np.sqrt(mse_value))

        history["iteration"].append(iteration)
        history["train_rmse"].append(train_rmse)
        history["loss"].append(loss_value)
        history["mse"].append(mse_value)
        history["stiffness_penalty"].append(penalty_value)
        history["energy_penalty"].append(energy_penalty_value)
        history["mean_offload_surrogate"].append(offload_surrogate)
        history["stiffness_change_penalty"].append(change_penalty_value)

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
    cubic = EXPERIMENT_CUBIC_RATIO / max(EXPERIMENT_CUBIC_REFERENCE_EXTENSION ** 2, 1e-12)
    return torch.sum(0.5 * stiffness * stretch**2 + 0.25 * stiffness * cubic * stretch**4)


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


def torch_torque_components_batch(
    topology, theta, stiffness, relax_internal, relaxation_steps, relaxation_lr
):
    """Return each spring's torque after jointly relaxing the network."""
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
    cubic = EXPERIMENT_CUBIC_RATIO / max(EXPERIMENT_CUBIC_REFERENCE_EXTENSION ** 2, 1e-12)
    force_on_a = stiffness * stretch + stiffness * cubic * stretch**3
    force_on_a = force_on_a.unsqueeze(2) * direction

    components = torch.zeros(
        (len(theta), len(a)), dtype=positions.dtype, device=positions.device
    )
    limb2_indices = set(int(index) for index in topology["limb2_indices"].detach().cpu().numpy())
    for spring_index in range(len(a)):
        node_a = int(a[spring_index])
        node_b = int(b[spring_index])
        if node_a in limb2_indices:
            r = positions[:, node_a, :]
            force = force_on_a[:, spring_index, :]
            components[:, spring_index] += r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
        if node_b in limb2_indices:
            r = positions[:, node_b, :]
            force = -force_on_a[:, spring_index, :]
            components[:, spring_index] += r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
    return components


def torch_torque_batch(topology, theta, stiffness, relax_internal, relaxation_steps, relaxation_lr):
    return torch.sum(
        torch_torque_components_batch(
            topology, theta, stiffness, relax_internal, relaxation_steps, relaxation_lr
        ),
        dim=1,
    )


def differentiable_relaxed_stiffness_torque(
    topology,
    theta,
    stiffness,
    relaxation_steps,
    step_size,
    max_step,
):
    """Torque with a small, differentiable unrolled equilibrium solve."""
    prescribed = torch_prescribed_positions(topology, theta)
    internal_indices = topology["internal_indices"]
    internal = prescribed[:, internal_indices, :]
    for _ in range(relaxation_steps):
        internal = internal.requires_grad_(True)
        positions = prescribed.clone()
        positions[:, internal_indices, :] = internal
        energy = torch_spring_energy(topology, positions, stiffness)
        gradient = torch.autograd.grad(energy, internal, create_graph=True)[0]
        force = -gradient
        displacement = max_step * torch.tanh(step_size * force / max(max_step, 1e-12))
        internal = internal + displacement

    positions = prescribed.clone()
    positions[:, internal_indices, :] = internal
    a, b = topology["spring_a"], topology["spring_b"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    direction = delta / length.unsqueeze(2)
    stretch = length - topology["rest_lengths"].unsqueeze(0)
    cubic = EXPERIMENT_CUBIC_RATIO / max(
        EXPERIMENT_CUBIC_REFERENCE_EXTENSION**2, 1e-12
    )
    force_on_a = (
        stiffness * stretch + stiffness * cubic * stretch**3
    ).unsqueeze(2) * direction
    torque = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)
    limb2 = set(
        int(index)
        for index in topology["limb2_indices"].detach().cpu().numpy()
    )
    for spring_index in range(len(a)):
        node_a, node_b = int(a[spring_index]), int(b[spring_index])
        if node_a in limb2:
            r, force = positions[:, node_a, :], force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
        if node_b in limb2:
            r, force = positions[:, node_b, :], -force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
    return torque


def profile_subset(dataset, profile_indices):
    """Select complete causal trajectories from a flattened dataset."""
    samples = dataset["samples_per_profile"]
    indices = np.asarray(profile_indices, dtype=int)
    flat = (indices[:, None] * samples + np.arange(samples)[None, :]).reshape(-1)
    result = dict(dataset)
    total_rows = len(dataset["target"])
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and value.shape[0] == total_rows:
            result[key] = value[flat]
    return result


def differentiable_mechanics_correction(
    model,
    dataset,
    topology_path,
    min_k,
    max_k,
    initial_k,
    updates,
    learning_rate,
    relaxation_steps,
    relaxation_step_size,
    relaxation_max_step,
    device,
    stiffness_weight=0.0,
    stiffness_change_weight=0.0,
    energy_weight=0.0,
    motoring_efficiency=DEFAULT_MOTORING_EFFICIENCY,
    regen_efficiency=DEFAULT_REGEN_EFFICIENCY,
):
    """Fine-tune on a small set using gradients through relaxed mechanics."""
    torch_device = torch.device(select_training_device(device))
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, torch_device)
    parameters = {
        name: torch.tensor(
            value, dtype=torch.float32, device=torch_device, requires_grad=True
        )
        for name, value in model.items()
    }
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    profiles = len(dataset["target"]) // dataset["samples_per_profile"]
    samples = dataset["samples_per_profile"]
    window = dataset["window_size"]
    motion = torch.as_tensor(
        dataset["features"].reshape(profiles, samples, -1),
        dtype=torch.float32,
        device=torch_device,
    )
    target = torch.as_tensor(
        dataset["target"].reshape(profiles, samples),
        dtype=torch.float32,
        device=torch_device,
    )
    theta = torch.as_tensor(
        dataset["theta"].reshape(profiles, samples),
        dtype=torch.float32,
        device=torch_device,
    )
    theta_dot = torch.as_tensor(
        dataset["theta_dot"].reshape(profiles, samples),
        dtype=torch.float32,
        device=torch_device,
    )
    update_mask = torch.as_tensor(
        dataset["update_mask"].reshape(profiles, samples),
        dtype=torch.bool,
        device=torch_device,
    )
    initial_k_tensor = torch.as_tensor(
        initial_k, dtype=torch.float32, device=torch_device
    )
    baseline_power_mean = torch.clamp(
        torch.mean(
            torch_energy_burden_power(
                target * theta_dot, motoring_efficiency, regen_efficiency
            )
        ),
        min=1e-9,
    )
    torque_scale = max(dataset["torque_scale"], 1e-9)

    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        history = torch.zeros(
            (profiles, window, 3), dtype=motion.dtype, device=torch_device
        )
        held = None
        previous_stiffness = None
        squared_error = 0.0
        for sample_index in range(samples):
            inputs = torch.cat(
                (motion[:, sample_index, :], history.reshape(profiles, -1)), dim=1
            )
            hidden = torch.tanh(inputs @ parameters["w1"] + parameters["b1"])
            logits = hidden @ parameters["w2"] + parameters["b2"]
            sig = torch.sigmoid(torch.clamp(logits, -50.0, 50.0))
            candidate = min_k + (max_k - min_k) * sig
            stiffness = (
                candidate
                if held is None
                else torch.where(
                    update_mask[:, sample_index].unsqueeze(1), candidate, held
                )
            )
            held = stiffness
            torque = differentiable_relaxed_stiffness_torque(
                topology,
                theta[:, sample_index],
                stiffness,
                relaxation_steps,
                relaxation_step_size,
                relaxation_max_step,
            )
            error = torque - target[:, sample_index]
            mse_step = torch.mean(error**2)
            loss_step = mse_step / samples
            stiffness_delta = (stiffness - initial_k_tensor) / torch.clamp(
                initial_k_tensor, min=1.0
            )
            loss_step = loss_step + stiffness_weight * torch.mean(
                stiffness_delta**2
            ) / samples
            assisted_power = torch_energy_burden_power(
                (target[:, sample_index] - torque) * theta_dot[:, sample_index],
                motoring_efficiency,
                regen_efficiency,
            )
            loss_step = loss_step + (
                energy_weight
                * mse_step.detach()
                * torch.mean(assisted_power)
                / baseline_power_mean
                / samples
            )
            if previous_stiffness is not None:
                normalized_change = (
                    stiffness - previous_stiffness
                ) / max(max_k - min_k, 1e-9)
                loss_step = loss_step + (
                    stiffness_change_weight
                    * mse_step.detach()
                    * torch.mean(normalized_change**2)
                    / max(samples - 1, 1)
                )
            loss_step.backward()
            squared_error += float(torch.sum(error.detach() ** 2).cpu())
            previous_stiffness = stiffness.detach()
            motor = target[:, sample_index] - torque.detach()
            realized = torch.stack(
                (target[:, sample_index], torque.detach(), motor), dim=1
            ) / torque_scale
            history = torch.cat(
                (history[:, 1:, :], realized.unsqueeze(1)), dim=1
            )
            held = held.detach()
        optimizer.step()
        rmse = np.sqrt(squared_error / (profiles * samples))
        print(
            f"  mechanics correction update {update:3d}/{updates} | "
            f"relaxed RMSE {rmse:8.3f} N*m"
        )
    return {
        name: value.detach().cpu().numpy().astype(float).copy()
        for name, value in parameters.items()
    }


def refresh_surrogate_basis(
    model,
    dataset,
    topology_path,
    min_k,
    max_k,
    device,
    relaxation_steps,
):
    """Rebuild the local stiffness basis at a controller's relaxed operating points.

    At the anchor schedule, ``sum(basis * stiffness)`` exactly reproduces the
    relaxed torque. Subsequent optimization treats the relaxed geometry as a
    local linearization until the next refresh.
    """
    selected_device = select_training_device(device)
    torch_device = torch.device(selected_device)
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, torch_device)
    samples = dataset["samples_per_profile"]
    profiles = len(dataset["target"]) // samples
    window_size = dataset["window_size"]
    torque_scale = max(dataset["torque_scale"], 1e-9)
    motion = dataset["features"].reshape(profiles, samples, -1)
    target = dataset["target"].reshape(profiles, samples)
    theta = dataset["theta"].reshape(profiles, samples)
    update_mask = dataset["update_mask"].reshape(profiles, samples)
    history = np.zeros((profiles, window_size, 3), dtype=float)
    refreshed = np.empty((profiles, samples, len(network.springs)), dtype=float)
    held_stiffness = None

    for sample_index in range(samples):
        inputs = np.hstack((motion[:, sample_index, :], history.reshape(profiles, -1)))
        candidate, _ = forward(model, inputs, min_k, max_k)
        if held_stiffness is None:
            stiffness = candidate
        else:
            stiffness = np.where(
                update_mask[:, sample_index, None], candidate, held_stiffness
            )
        held_stiffness = stiffness
        theta_tensor = torch.as_tensor(
            theta[:, sample_index], dtype=torch.float32, device=torch_device
        )
        stiffness_tensor = torch.as_tensor(
            stiffness, dtype=torch.float32, device=torch_device
        )
        components = torch_torque_components_batch(
            topology,
            theta_tensor,
            stiffness_tensor,
            relax_internal=True,
            relaxation_steps=relaxation_steps,
            relaxation_lr=0.03,
        ).detach().cpu().numpy()
        basis = components / np.maximum(stiffness, 1e-9)
        refreshed[:, sample_index, :] = basis
        spring_torque = np.sum(basis * stiffness, axis=1)
        motor_torque = target[:, sample_index] - spring_torque
        realized = np.stack(
            (target[:, sample_index], spring_torque, motor_torque), axis=1
        ) / torque_scale
        history = np.concatenate((history[:, 1:, :], realized[:, None, :]), axis=1)

    updated = dict(dataset)
    updated["basis"] = refreshed.reshape(-1, refreshed.shape[2])
    return updated


def model_uses_torque_history(model, dataset):
    expected = dataset["features"].shape[1] + 3 * dataset["window_size"]
    return model["w1"].shape[0] == expected


def recurrent_dataset_with_mechanics(
    model,
    dataset,
    topology_path,
    min_k,
    max_k,
    relax_internal,
    mechanics_backend,
    device,
    relaxation_steps,
):
    """Replay profiles causally using previously realized target/spring/motor torque."""
    samples = dataset["samples_per_profile"]
    profile_count = len(dataset["theta"]) // samples
    window_size = dataset["window_size"]
    torque_scale = max(dataset["torque_scale"], 1e-9)
    motion = dataset["features"].reshape(profile_count, samples, -1)
    target = dataset["target"].reshape(profile_count, samples)
    theta = dataset["theta"].reshape(profile_count, samples)
    history = np.zeros((profile_count, window_size, 3), dtype=float)
    predicted = np.empty((profile_count, samples), dtype=float)
    stiffness = np.empty((profile_count, samples, len(load_network(topology_path)[0].springs)), dtype=float)
    selected_backend = select_mechanics_backend(mechanics_backend)
    update_mask = dataset["update_mask"].reshape(profile_count, samples)
    held_stiffness = None

    if selected_backend == "torch":
        selected_device = select_training_device(device)
        torch_device = torch.device(selected_device)
        network, _ = load_network(topology_path)
        topology = torch_topology_data(network, torch_device)
    else:
        networks = [load_network(topology_path)[0] for _ in range(profile_count)]

    for sample_index in range(samples):
        inputs = np.hstack((motion[:, sample_index, :], history.reshape(profile_count, -1)))
        candidate_stiffness, _ = forward(model, inputs, min_k, max_k)
        if held_stiffness is None:
            stiffness_step = candidate_stiffness
        else:
            stiffness_step = np.where(
                update_mask[:, sample_index, None], candidate_stiffness, held_stiffness
            )
        held_stiffness = stiffness_step
        stiffness[:, sample_index, :] = stiffness_step

        if selected_backend == "torch":
            theta_step = torch.as_tensor(theta[:, sample_index], dtype=torch.float32, device=torch_device)
            stiffness_tensor = torch.as_tensor(stiffness_step, dtype=torch.float32, device=torch_device)
            torque_step = torch_torque_batch(
                topology,
                theta_step,
                stiffness_tensor,
                relax_internal=relax_internal,
                relaxation_steps=relaxation_steps,
                relaxation_lr=0.03,
            ).detach().cpu().numpy()
        else:
            torque_step = np.empty(profile_count, dtype=float)
            for profile_index, network in enumerate(networks):
                for spring, value in zip(network.springs, stiffness_step[profile_index]):
                    spring.stiffness_k = float(value)
                _, _, torque_step[profile_index] = network.evaluate(
                    float(theta[profile_index, sample_index]), relax_internal=relax_internal
                )

        predicted[:, sample_index] = torque_step
        motor_step = target[:, sample_index] - torque_step
        realized = np.stack((target[:, sample_index], torque_step, motor_step), axis=1) / torque_scale
        history = np.concatenate((history[:, 1:, :], realized[:, None, :]), axis=1)

    return predicted.reshape(-1), stiffness.reshape(-1, stiffness.shape[2])


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
    if model_uses_torque_history(model, dataset):
        return recurrent_dataset_with_mechanics(
            model,
            dataset,
            topology_path,
            min_k,
            max_k,
            relax_internal=True,
            mechanics_backend=mechanics_backend,
            device=device,
            relaxation_steps=relaxation_steps,
        )
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
    if model_uses_torque_history(model, dataset):
        return recurrent_dataset_with_mechanics(
            model,
            dataset,
            topology_path,
            min_k,
            max_k,
            relax_internal=relax_internal,
            mechanics_backend=mechanics_backend,
            device=device,
            relaxation_steps=relaxation_steps,
        )
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


def energy_accounting_metrics(
    t,
    theta_dot,
    target,
    predicted,
    motoring_efficiency=DEFAULT_MOTORING_EFFICIENCY,
    regen_efficiency=DEFAULT_REGEN_EFFICIENCY,
):
    residual = target - predicted
    baseline = numpy_power_accounting(
        target * theta_dot, motoring_efficiency, regen_efficiency
    )
    assisted = numpy_power_accounting(
        residual * theta_dot, motoring_efficiency, regen_efficiency
    )

    def energy(accounting, name):
        return float(integrate_trapezoid(accounting[name], t))

    baseline_burden = energy(baseline, "energy_burden_power")
    assisted_burden = energy(assisted, "energy_burden_power")
    offload = 0.0
    if abs(baseline_burden) >= 1e-12:
        offload = 100.0 * (baseline_burden - assisted_burden) / baseline_burden
    return {
        "offload_pct": offload,
        "baseline_energy_burden_j": baseline_burden,
        "energy_burden_with_spring_j": assisted_burden,
        "net_battery_energy_with_spring_j": energy(assisted, "net_battery_power"),
        "braking_energy_with_spring_j": energy(assisted, "braking_mechanical_power"),
        "regenerated_energy_with_spring_j": energy(assisted, "regenerated_power"),
    }


def summarize_profiles(
    profile_params,
    dataset,
    predicted,
    motoring_efficiency=DEFAULT_MOTORING_EFFICIENCY,
    regen_efficiency=DEFAULT_REGEN_EFFICIENCY,
):
    rows = []
    samples = dataset["samples_per_profile"]
    for profile_index, params in enumerate(profile_params):
        start = profile_index * samples
        stop = start + samples
        target = dataset["target"][start:stop]
        pred = predicted[start:stop]
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        energy_metrics = energy_accounting_metrics(
            dataset["t"][start:stop],
            dataset["theta_dot"][start:stop],
            target,
            pred,
            motoring_efficiency,
            regen_efficiency,
        )
        rows.append(
            {
                "profile": params["name"],
                "family": params["family"],
                "roughness_score": float(params.get("roughness_score", np.nan)),
                "rmse_nm": rmse,
                **energy_metrics,
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
    columns = [
        "profile",
        "family",
        "roughness_score",
        "rmse_nm",
        "offload_pct",
        "baseline_energy_burden_j",
        "energy_burden_with_spring_j",
        "net_battery_energy_with_spring_j",
        "braking_energy_with_spring_j",
        "regenerated_energy_with_spring_j",
        "mean_abs_residual_nm",
        "peak_abs_residual_nm",
    ]
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


def plot_test_examples(path, model, test_params, angles_rad, basis_by_angle, duration, samples, window_size, scales, min_k, max_k, seed, topology_path, stiffness_update_mode="timestep", include_profile_descriptor=True, motion_mode="randomized", fixed_frequency_hz=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(14, 15), constrained_layout=True)
    axes = axes.ravel()
    for profile_index, params in enumerate(test_params[:6]):
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
            stiffness_update_mode=stiffness_update_mode,
            include_profile_descriptor=include_profile_descriptor,
            motion_mode=motion_mode,
            fixed_frequency_hz=fixed_frequency_hz,
        )
        predicted, _ = predict_dataset_relaxed(
            model,
            dataset,
            topology_path,
            min_k,
            max_k,
            mechanics_backend="scipy",
        )
        time_ax = axes[profile_index]
        angle_ax = axes[profile_index + 6]
        residual = dataset["target"] - predicted
        combined = predicted + residual
        time_ax.plot(dataset["t"], combined, color="tab:green", linestyle=":", linewidth=3.0, label="spring + motor", zorder=1)
        time_ax.plot(dataset["t"], predicted, color="tab:blue", linewidth=2.5, label="spring", zorder=3)
        time_ax.plot(dataset["t"], residual, color="tab:red", linestyle="-.", linewidth=2.2, label="residual motor", zorder=3)
        time_ax.plot(dataset["t"], dataset["target"], color="black", linestyle="--", linewidth=2.0, label="target", zorder=4)
        time_ax.set_title(f"{params['family']} / {params['name']}")
        time_ax.set_xlabel("time [s]")
        time_ax.set_ylabel("torque [N*m]")
        angle_order = np.argsort(dataset["theta"])
        angle_deg = np.rad2deg(dataset["theta"])
        angle_ax.scatter(angle_deg, predicted, s=18, color="tab:blue", alpha=0.8, label="spring", zorder=3)
        angle_ax.scatter(angle_deg, residual, s=16, color="tab:red", marker="x", alpha=0.75, label="residual motor", zorder=3)
        angle_ax.plot(angle_deg[angle_order], dataset["target"][angle_order], color="black", linestyle="--", linewidth=2.0, label="target", zorder=4)
        angle_ax.set_xlabel("joint angle [deg]")
        angle_ax.set_ylabel("torque [N*m]")
        for ax in (time_ax, angle_ax):
            ax.axhline(0.0, color="0.7", linewidth=1.0)
            ax.grid(True, alpha=0.25)
    axes[0].legend()
    axes[6].legend()
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
    global EXPERIMENT_CUBIC_RATIO, EXPERIMENT_CUBIC_REFERENCE_EXTENSION
    parser = argparse.ArgumentParser(description="Train adaptive spring stiffnesses on generated piecewise-linear torque profiles.")
    parser.add_argument(
        "--network",
        choices=sorted(TRAINING_NETWORK_PRESETS),
        default="fan",
        help="Topology preset to train (default: fan).",
    )
    parser.add_argument("--topology", default=None, help="Custom topology JSON; overrides the network preset.")
    parser.add_argument(
        "--profiles-per-family",
        type=int,
        default=2000,
        help="Arbitrary training profiles in each relative shape-roughness class.",
    )
    parser.add_argument(
        "--classification-mode",
        choices=["roughness", "periodicity-high", "periodicity-medium", "periodicity-low"],
        default="roughness",
        help=(
            "Dataset selection method. Periodicity modes rank generated trajectories by "
            "cycle repeatability and train only on the selected third."
        ),
    )
    parser.add_argument(
        "--test-profiles-per-family",
        type=int,
        default=400,
        help="Held-out arbitrary profiles in each relative shape-roughness class.",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Trajectory duration in seconds.")
    parser.add_argument("--samples", type=int, default=160, help="Samples per generated trajectory.")
    parser.add_argument(
        "--motion-mode",
        choices=["randomized", "triangular"],
        default="randomized",
        help="Joint motion generator. Triangular repeatedly sweeps from -45 to +45 degrees and back.",
    )
    parser.add_argument(
        "--fixed-frequency-hz",
        type=float,
        default=None,
        help="Use one constant motion frequency for every trajectory instead of each profile's random frequency.",
    )
    parser.add_argument("--window-size", type=int, default=10, help="Recent motion samples used as neural-network input.")
    parser.add_argument(
        "--include-profile-descriptor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include global torque-profile knots (privileged information). Disabled by default for causal training.",
    )
    parser.add_argument(
        "--stiffness-update-mode",
        choices=["timestep", "period"],
        default="timestep",
        help="Update stiffness every sample or only at nominal gait-period boundaries.",
    )
    parser.add_argument("--output-name", default=None, help="Output stem; defaults to a topology-specific model name.")
    parser.add_argument(
        "--write-torque-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the very large per-timestep torque CSV. Disabled by default.",
    )
    parser.add_argument("--iterations", type=int, default=5000, help="Gradient descent iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam step size.")
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Weight optimizer. 'sgd' is plain full-batch gradient descent with no momentum.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden units in the neural stiffness model.")
    parser.add_argument("--min-stiffness", type=float, default=1.0, help="Minimum learned stiffness in N/m.")
    parser.add_argument("--max-stiffness", type=float, default=800.0, help="Maximum learned stiffness in N/m.")
    parser.add_argument(
        "--stiffness-weight",
        type=float,
        default=0.0,
        help="Optional penalty for moving far from baseline stiffnesses (default: disabled).",
    )
    parser.add_argument(
        "--stiffness-change-weight",
        type=float,
        default=0.0,
        help="Soft penalty on normalized timestep-to-timestep stiffness changes.",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.35,
        help="Weight for reducing bidirectional motor energy burden during training.",
    )
    parser.add_argument(
        "--motoring-efficiency",
        type=float,
        default=DEFAULT_MOTORING_EFFICIENCY,
        help="Motor/drive efficiency while delivering positive shaft power (default: 0.85).",
    )
    parser.add_argument(
        "--regen-efficiency",
        type=float,
        default=DEFAULT_REGEN_EFFICIENCY,
        help="Fraction of mechanical braking energy returned electrically (default: 0.60).",
    )
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
        default="torch",
        help="Mechanics backend for baseline/adaptive evaluation (default: torch). Use scipy explicitly as a fallback.",
    )
    parser.add_argument("--mechanics-batch-size", type=int, default=8192, help="Samples per GPU mechanics batch.")
    parser.add_argument("--relaxation-steps", type=int, default=80, help="PyTorch internal-node relaxation steps per mechanics batch.")
    parser.add_argument(
        "--surrogate-refreshes",
        type=int,
        default=0,
        help=(
            "Number of times to replay relaxed mechanics and rebuild the local "
            "per-spring torque basis during training (default: 0)."
        ),
    )
    parser.add_argument(
        "--mechanics-correction-phases",
        type=int,
        default=0,
        help=(
            "Preliminary hybrid mode: number of short fine-tuning phases that "
            "backpropagate through unrolled relaxed mechanics (default: 0)."
        ),
    )
    parser.add_argument("--mechanics-correction-profiles", type=int, default=8)
    parser.add_argument("--mechanics-correction-updates", type=int, default=10)
    parser.add_argument("--mechanics-correction-relaxation-steps", type=int, default=40)
    parser.add_argument("--mechanics-correction-learning-rate", type=float, default=0.0003)
    parser.add_argument("--mechanics-correction-step-size", type=float, default=0.0001)
    parser.add_argument("--mechanics-correction-max-step-mm", type=float, default=5.0)
    parser.add_argument(
        "--cubic-ratio", type=float, default=0.0,
        help="Experimental dimensionless cubic hardening at the reference extension.",
    )
    parser.add_argument("--cubic-reference-extension-mm", type=float, default=50.0)
    parser.add_argument("--cubic-design-extension-mm", type=float, default=1000.0)
    parser.add_argument("--cubic-min-tangent-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    args = parser.parse_args()
    if args.surrogate_refreshes < 0:
        parser.error("--surrogate-refreshes cannot be negative")
    if args.surrogate_refreshes >= args.iterations:
        parser.error("--surrogate-refreshes must be smaller than --iterations")
    if args.mechanics_correction_phases < 0:
        parser.error("--mechanics-correction-phases cannot be negative")
    if min(
        args.mechanics_correction_profiles,
        args.mechanics_correction_updates,
        args.mechanics_correction_relaxation_steps,
    ) <= 0:
        parser.error("Mechanics-correction profiles, updates, and relaxation steps must be positive")
    if args.mechanics_correction_learning_rate <= 0.0:
        parser.error("--mechanics-correction-learning-rate must be positive")
    if args.mechanics_correction_step_size <= 0.0 or args.mechanics_correction_max_step_mm <= 0.0:
        parser.error("Mechanics-correction step sizes must be positive")
    if args.fixed_frequency_hz is not None and args.fixed_frequency_hz <= 0.0:
        parser.error("--fixed-frequency-hz must be positive")
    validate_efficiencies(args.motoring_efficiency, args.regen_efficiency)
    preset = TRAINING_NETWORK_PRESETS[args.network]
    if args.topology is None:
        args.topology = preset["topology"]
    if args.output_name is None:
        args.output_name = preset["output_name"]
    mechanics_backend = select_mechanics_backend(args.mechanics_backend)

    network, topology = load_network(args.topology)
    angles_rad = np.radians(ANGLE_DEGREES)
    if args.cubic_reference_extension_mm <= 0.0 or args.cubic_design_extension_mm <= 0.0:
        parser.error("Cubic reference and design extensions must be positive")
    tangent_ratio = 1.0 + 3.0 * args.cubic_ratio * (
        args.cubic_design_extension_mm / args.cubic_reference_extension_mm
    ) ** 2
    if tangent_ratio < args.cubic_min_tangent_ratio:
        parser.error(
            "Unsafe cubic softening: tangent stiffness would fall to "
            f"{tangent_ratio:.3f} times linear stiffness at the design extension."
        )
    EXPERIMENT_CUBIC_RATIO = args.cubic_ratio
    EXPERIMENT_CUBIC_REFERENCE_EXTENSION = args.cubic_reference_extension_mm / 1000.0
    basis_by_angle = spring_torque_basis(
        network, angles_rad, relax_internal=True,
        cubic_ratio=args.cubic_ratio,
        cubic_reference_extension=args.cubic_reference_extension_mm / 1000.0,
    )
    base_k = initial_stiffnesses(network)

    rng = np.random.default_rng(args.seed)
    if args.classification_mode.startswith("periodicity-"):
        periodicity_class = args.classification_mode.removeprefix("periodicity-")
        train_params = generate_periodicity_profiles(
            rng,
            args.profiles_per_family,
            args.duration,
            args.samples,
            args.seed + 1_000,
            periodicity_class,
        )
        test_params = generate_periodicity_profiles(
            rng,
            args.test_profiles_per_family,
            args.duration,
            args.samples,
            args.seed + 2_000,
            periodicity_class,
        )
        active_classification = PERIODICITY_CLASSIFICATION
        active_families = (f"{periodicity_class}_periodicity",)
    else:
        train_params = generate_classified_profile_parameters(rng, args.profiles_per_family)
        test_params = generate_classified_profile_parameters(rng, args.test_profiles_per_family)
        active_classification = PROFILE_CLASSIFICATION
        active_families = TERRAIN_FAMILIES

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
    train_dataset = build_dataset(
        train_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.seed + 20_000,
        stiffness_update_mode=args.stiffness_update_mode,
        progress_label="training dataset profiles",
        progress_interval=args.progress_interval,
        include_profile_descriptor=args.include_profile_descriptor,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
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
        stiffness_update_mode=args.stiffness_update_mode,
        progress_label="test dataset profiles",
        progress_interval=args.progress_interval,
        include_profile_descriptor=args.include_profile_descriptor,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
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
    print("Profile set: arbitrary five-knot torque profiles")
    print(f"Classification: {active_classification}")
    print(f"Selected classes: {', '.join(active_families)}")
    print(f"Profiles per class: train {args.profiles_per_family} | test {args.test_profiles_per_family}")
    print(f"Training trajectories: {len(train_params)} | test trajectories: {len(test_params)}")
    print(f"Samples per trajectory: {args.samples} | motion window: {args.window_size} samples")
    print(f"Motion mode: {args.motion_mode}")
    print(
        f"Motion frequency: {args.fixed_frequency_hz:g} Hz (fixed across all trajectories)"
        if args.fixed_frequency_hz is not None
        else "Motion frequency: randomized per trajectory"
    )
    print(f"Stiffness update mode: {args.stiffness_update_mode}")
    print(f"Stiffness change weight: {args.stiffness_change_weight:g}")
    print(f"Optimizer: {args.optimizer} | learning rate: {args.learning_rate:g}")
    descriptor_description = "10 normalized profile knots + " if args.include_profile_descriptor else ""
    feature_description = (
        descriptor_description
        + f"{args.window_size} * theta/theta_dot/theta_ddot + "
        + f"{args.window_size} * previous target/spring/motor torque"
    )
    feature_count = train_dataset["features"].shape[1] + 3 * args.window_size
    print(
        "Input mode: commanded profile plus causal motion and realized torque history"
        if args.include_profile_descriptor
        else "Input mode: strictly causal motion and realized torque history (no profile descriptor)"
    )
    print(f"Feature count: {feature_count} ({feature_description})")
    print(
        "Training torque basis: relaxed baseline geometry"
        if not args.surrogate_refreshes
        else (
            "Training torque basis: relaxed baseline geometry + "
            f"{args.surrogate_refreshes} controller operating-point refresh(es)"
        )
    )
    print(f"Reported metrics: full relaxed network evaluation via {mechanics_backend}")
    print(
        f"Energy accounting: motoring efficiency {args.motoring_efficiency:.2f} | "
        f"regeneration efficiency {args.regen_efficiency:.2f}"
    )
    print(f"Fixed-stiffness baseline train RMSE: {train_baseline_rmse:.4f} N*m")
    print(f"Fixed-stiffness baseline test RMSE:  {test_baseline_rmse:.4f} N*m")

    phase_count = args.surrogate_refreshes + 1
    if args.mechanics_correction_phases > phase_count:
        parser.error(
            "--mechanics-correction-phases cannot exceed the number of "
            "surrogate phases (--surrogate-refreshes + 1)"
        )
    phase_iterations = [
        args.iterations // phase_count + (1 if index < args.iterations % phase_count else 0)
        for index in range(phase_count)
    ]
    active_train_dataset = train_dataset
    model = None
    training_history = None
    completed_iterations = 0
    correction_phase_indices = set()
    correction_dataset = None
    if args.mechanics_correction_phases:
        correction_phase_indices = set(
            np.linspace(
                0,
                phase_count - 1,
                args.mechanics_correction_phases,
                dtype=int,
            ).tolist()
        )
        correction_rng = np.random.default_rng(args.seed + 90_000)
        correction_count = min(
            args.mechanics_correction_profiles, len(train_params)
        )
        correction_profiles = correction_rng.choice(
            len(train_params), size=correction_count, replace=False
        )
        correction_dataset = profile_subset(
            train_dataset, correction_profiles
        )
        print(
            "Preliminary differentiable-mechanics correction: "
            f"{len(correction_phase_indices)} phases | "
            f"{correction_count} complete profiles | "
            f"{args.mechanics_correction_updates} updates per phase"
        )
    for phase_index, iteration_count in enumerate(phase_iterations):
        if phase_index:
            print(
                f"Refreshing surrogate at controller operating points "
                f"({phase_index}/{args.surrogate_refreshes})..."
            )
            active_train_dataset = refresh_surrogate_basis(
                model,
                train_dataset,
                args.topology,
                args.min_stiffness,
                args.max_stiffness,
                args.device,
                args.relaxation_steps,
            )
        phase_model, phase_history = train_model(
            dataset=active_train_dataset,
            initial_k=base_k,
            hidden_dim=args.hidden_dim,
            iterations=iteration_count,
            learning_rate=args.learning_rate,
            min_k=args.min_stiffness,
            max_k=args.max_stiffness,
            stiffness_weight=args.stiffness_weight,
            seed=args.seed,
            progress_interval=args.progress_interval,
            device=args.device,
            energy_weight=args.energy_weight,
            motoring_efficiency=args.motoring_efficiency,
            regen_efficiency=args.regen_efficiency,
            stiffness_change_weight=args.stiffness_change_weight,
            optimizer_name=args.optimizer,
            initial_model=model,
        )
        model = phase_model
        phase_history["iteration"] = [
            completed_iterations + value for value in phase_history["iteration"]
        ]
        completed_iterations += iteration_count
        if training_history is None:
            training_history = phase_history
        else:
            for key, values in phase_history.items():
                training_history[key].extend(values)
        if phase_index in correction_phase_indices:
            correction_number = (
                sorted(correction_phase_indices).index(phase_index) + 1
            )
            print(
                "Running differentiable-mechanics correction "
                f"({correction_number}/{len(correction_phase_indices)})..."
            )
            model = differentiable_mechanics_correction(
                model,
                correction_dataset,
                args.topology,
                args.min_stiffness,
                args.max_stiffness,
                base_k,
                args.mechanics_correction_updates,
                args.mechanics_correction_learning_rate,
                args.mechanics_correction_relaxation_steps,
                args.mechanics_correction_step_size,
                args.mechanics_correction_max_step_mm / 1000.0,
                args.device,
                stiffness_weight=args.stiffness_weight,
                stiffness_change_weight=args.stiffness_change_weight,
                energy_weight=args.energy_weight,
                motoring_efficiency=args.motoring_efficiency,
                regen_efficiency=args.regen_efficiency,
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
    test_schedule = test_stiffness.reshape(len(test_params), args.samples, -1)
    test_changes = np.abs(np.diff(test_schedule, axis=1))
    print(
        f"Held-out stiffness activity: mean |dk| {np.mean(test_changes):.4f} N/m per step | "
        f"max |dk| {np.max(test_changes):.4f} N/m"
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
    energy_args = (args.motoring_efficiency, args.regen_efficiency)
    train_rows = summarize_profiles(train_params, train_dataset, train_pred, *energy_args)
    test_rows = summarize_profiles(test_params, test_dataset, test_pred, *energy_args)
    baseline_relaxed_test_rows = summarize_profiles(
        test_params, test_dataset, baseline_fixed_test, *energy_args
    )
    baseline_unrelaxed_test_rows = summarize_profiles(
        test_params, test_dataset, baseline_unrelaxed_test, *energy_args
    )
    adaptive_unrelaxed_test_rows = summarize_profiles(
        test_params, test_dataset, adaptive_unrelaxed_test, *energy_args
    )
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
    model_path = output_dir / "models" / "adaptive_stiffness" / f"{output_name}.npz"
    table_dir = output_dir / "tables" / "adaptive_stiffness"
    plot_dir = output_dir / "plots" / "adaptive_stiffness" / "dataset_examples"
    train_table_path = table_dir / f"{output_name}_train_results.csv"
    test_table_path = table_dir / f"{output_name}_test_results.csv"
    torque_trace_path = table_dir / f"{output_name}_test_torque_trace.csv"
    model_comparison_path = table_dir / f"{output_name}_mechanics_comparison.csv"
    figure_path = plot_dir / f"{output_name}_test_examples.png"
    convergence_path = plot_dir / f"{output_name}_training_convergence.png"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    save_model(
        model_path,
        model,
        output_name,
        args.min_stiffness,
        args.max_stiffness,
        feature_type=(
            "profile_motion_torque_window"
            if args.include_profile_descriptor
            else "causal_motion_torque_window"
        ),
        window_size=args.window_size,
        theta_scale=scales["theta"],
        theta_dot_scale=scales["theta_dot"],
        theta_ddot_scale=scales["theta_ddot"],
        torque_scale=scales["torque"],
        training_device=select_training_device(args.device),
        hidden_dim=args.hidden_dim,
        energy_weight=args.energy_weight,
        motoring_efficiency=args.motoring_efficiency,
        regen_efficiency=args.regen_efficiency,
        profile_set=(
            f"{periodicity_class}_periodicity_piecewise"
            if args.classification_mode.startswith("periodicity-")
            else "classified_arbitrary"
        ),
        profile_classification=active_classification,
        classification_mode=args.classification_mode,
        profiles_per_family=args.profiles_per_family,
        test_profiles_per_family=args.test_profiles_per_family,
        mechanics_backend=mechanics_backend,
        mechanics_batch_size=args.mechanics_batch_size,
        relaxation_steps=args.relaxation_steps,
        surrogate_refreshes=args.surrogate_refreshes,
        cubic_ratio=args.cubic_ratio,
        cubic_reference_extension_mm=args.cubic_reference_extension_mm,
        cubic_design_extension_mm=args.cubic_design_extension_mm,
        cubic_min_tangent_ratio=args.cubic_min_tangent_ratio,
        seed=args.seed,
        mechanics_correction_phases=args.mechanics_correction_phases,
        mechanics_correction_profiles=args.mechanics_correction_profiles,
        mechanics_correction_updates=args.mechanics_correction_updates,
        mechanics_correction_relaxation_steps=args.mechanics_correction_relaxation_steps,
        mechanics_correction_learning_rate=args.mechanics_correction_learning_rate,
        mechanics_correction_step_size=args.mechanics_correction_step_size,
        mechanics_correction_max_step_mm=args.mechanics_correction_max_step_mm,
        duration=args.duration,
        samples=args.samples,
        stiffness_update_mode=args.stiffness_update_mode,
        stiffness_change_weight=args.stiffness_change_weight,
        include_profile_descriptor=args.include_profile_descriptor,
        optimizer=args.optimizer,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
    )
    write_profile_rows(train_table_path, train_rows)
    write_profile_rows(test_table_path, test_rows)
    if args.write_torque_trace:
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
        stiffness_update_mode=args.stiffness_update_mode,
        include_profile_descriptor=args.include_profile_descriptor,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
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
    if args.write_torque_trace:
        print(f"Saved test torque trace to {torque_trace_path}")
    else:
        print("Skipped large test torque trace (enable with --write-torque-trace).")
    print(f"Saved mechanics comparison to {model_comparison_path}")
    print(f"Saved test example plot to {figure_path}")
    print(f"Saved training convergence plot to {convergence_path}")
    plt.show()


if __name__ == "__main__":
    main()
