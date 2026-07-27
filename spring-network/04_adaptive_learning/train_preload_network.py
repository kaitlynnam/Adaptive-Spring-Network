"""Train a causal neural controller that changes spring preload, not stiffness.

This standalone proof-of-concept keeps each spring's material stiffness fixed
and changes coordinated preload modes around a neutral operating point.
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

from profile_generator import (
    ANGLE_LIMIT_RAD,
    generate_classified_profile_parameters,
)
from energy_accounting import (
    DEFAULT_MOTORING_EFFICIENCY,
    DEFAULT_REGEN_EFFICIENCY,
    torch_energy_burden_power,
    validate_efficiencies,
)
from topology_loader import load_network
from train_adaptive_dataset import (
    generate_motion_trajectory,
    generate_periodicity_profiles,
    motion_window_features,
    normalization_scales,
    torch_prescribed_positions,
    torch_topology_data,
)


def torque_curve(
    topology_path, angles, spring_index=None, preload_delta=0.0, reference_preload=0.0
):
    network, _ = load_network(topology_path)
    for spring in network.springs:
        spring.rest_length = max(spring.rest_length - reference_preload, 0.005)
    if spring_index is not None:
        network.springs[spring_index].rest_length = max(
            network.springs[spring_index].rest_length - preload_delta, 0.005
        )
    values = []
    for theta in angles:
        _, _, torque = network.evaluate(float(theta), relax_internal=True)
        values.append(torque)
    return np.asarray(values, dtype=float)


def preload_mechanics(topology_path, angles, finite_difference, reference_preload=0.0):
    network, _ = load_network(topology_path)
    base = torque_curve(topology_path, angles, reference_preload=reference_preload)
    sensitivity = np.empty((len(angles), len(network.springs)), dtype=float)
    for spring_index in range(len(network.springs)):
        perturbed = torque_curve(
            topology_path,
            angles,
            spring_index,
            finite_difference,
            reference_preload=reference_preload,
        )
        sensitivity[:, spring_index] = (perturbed - base) / finite_difference
    return base, sensitivity


def interpolate_columns(values, grid, theta):
    if values.ndim == 1:
        return np.interp(theta, grid, values)
    return np.column_stack([np.interp(theta, grid, values[:, i]) for i in range(values.shape[1])])


def build_dataset(
    params, duration, samples, seed, angles, base_curve, sensitivity,
    window_size, scales, motion_mode="randomized", fixed_frequency_hz=None,
):
    features, target, theta_rows, theta_dot_rows, base_rows, sensitivity_rows, profile_index = [], [], [], [], [], [], []
    for index, profile in enumerate(params):
        t, theta, theta_dot, theta_ddot, torque = generate_motion_trajectory(
            profile, duration, samples, seed + index,
            motion_mode=motion_mode, fixed_frequency_hz=fixed_frequency_hz,
        )
        features.append(motion_window_features(theta, theta_dot, theta_ddot, window_size, scales))
        target.append(torque)
        theta_rows.append(theta)
        theta_dot_rows.append(theta_dot)
        base_rows.append(interpolate_columns(base_curve, angles, theta))
        sensitivity_rows.append(interpolate_columns(sensitivity, angles, theta))
        profile_index.append(np.full(samples, index, dtype=int))
    return {
        "features": np.vstack(features).astype(np.float32),
        "target": np.concatenate(target).astype(np.float32),
        "theta": np.concatenate(theta_rows).astype(np.float32),
        "theta_dot": np.concatenate(theta_dot_rows).astype(np.float32),
        "base": np.concatenate(base_rows).astype(np.float32),
        "sensitivity": np.vstack(sensitivity_rows).astype(np.float32),
        "reference_preload": None,
        "operating_length": None,
        "profile_index": np.concatenate(profile_index),
        "samples": samples,
        "profiles": len(params),
        "duration": float(duration),
        "window_size": int(window_size),
        "torque_scale": float(scales["torque"]),
    }


class PreloadController(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, group_assignment, max_preload, initial_preload, reference_preload):
        super().__init__()
        group_assignment = torch.as_tensor(group_assignment, dtype=torch.long)
        group_count = int(group_assignment.max().item()) + 1
        self.register_buffer("group_assignment", group_assignment)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, group_count),
        )
        max_preload = torch.as_tensor(max_preload, dtype=torch.float32)
        if max_preload.ndim == 0:
            max_preload = max_preload.repeat(group_count)
        self.register_buffer("max_preload", max_preload)
        initial_preload = torch.as_tensor(initial_preload, dtype=torch.float32)
        initial_preload = torch.minimum(torch.maximum(initial_preload, torch.zeros_like(max_preload)), max_preload)
        self.register_buffer("initial_preload", initial_preload)
        reference_preload = torch.as_tensor(reference_preload, dtype=torch.float32)
        self.register_buffer("neutral_preload", reference_preload[group_assignment])
        output_layer = self.net[-1]
        with torch.no_grad():
            torch.nn.init.normal_(output_layer.weight, mean=0.0, std=0.005)
            output_layer.bias.zero_()

    def forward(self, features):
        # A signed adaptive residual around the best fixed preload. Unlike a
        # sigmoid initialized at 0 or max, this retains a useful gradient when
        # the fixed optimum lies on a bound.
        residual = self.max_preload * torch.tanh(self.net(features))
        grouped = torch.clamp(self.initial_preload + residual, min=0.0)
        grouped = torch.minimum(grouped, self.max_preload)
        return grouped[:, self.group_assignment]


def preload_groups(network, mode="four"):
    """Assign springs to upper/lower and left/right coordinated modes."""
    if mode == "per-spring":
        return np.arange(len(network.springs), dtype=int)
    x_parts = {"three": 2, "four": 2, "six": 3, "eight": 4}[mode]
    midpoints = []
    for spring in network.springs:
        a = network.nodes[spring.node_a].local_position
        b = network.nodes[spring.node_b].local_position
        midpoint = 0.5 * (a + b)
        midpoints.append(midpoint)
    x_values = np.asarray([p[0] for p in midpoints])
    if x_parts == 1:
        x_group = np.zeros(len(midpoints), dtype=int)
    else:
        edges = np.quantile(x_values, np.linspace(0, 1, x_parts + 1)[1:-1])
        x_group = np.digitize(x_values, edges)
    groups = []
    for index, midpoint in enumerate(midpoints):
        upper = midpoint[1] >= 0.0
        if mode == "three":
            groups.append(int(x_group[index]) if upper else 2)
        else:
            groups.append((0 if upper else x_parts) + int(x_group[index]))
    return np.asarray(groups, dtype=int)


def tensors(dataset, device):
    tensor_keys = ("features", "target", "theta", "theta_dot", "base", "sensitivity")
    optional_keys = ("reference_preload", "operating_length")
    result = {
        key: torch.as_tensor(dataset[key], device=device)
        for key in tensor_keys
    }
    for key in optional_keys:
        if dataset.get(key) is not None:
            result[key] = torch.as_tensor(dataset[key], dtype=torch.float32, device=device)
    result["profiles"] = dataset["profiles"]
    result["samples"] = dataset["samples"]
    result["window_size"] = dataset["window_size"]
    result["torque_scale"] = dataset["torque_scale"]
    return result


def profile_batch(data, profile_indices):
    """Select complete trajectories while preserving the causal time layout."""
    indices = torch.as_tensor(profile_indices, dtype=torch.long, device=data["target"].device)
    samples = data["samples"]
    flat = (indices[:, None] * samples + torch.arange(samples, device=indices.device)[None, :]).reshape(-1)
    result = {
        key: value[flat]
        for key, value in data.items()
        if torch.is_tensor(value) and value.shape[0] == data["profiles"] * samples
    }
    result.update({
        "profiles": len(indices),
        "samples": samples,
        "window_size": data["window_size"],
        "torque_scale": data["torque_scale"],
    })
    return result


def predict(model, data, motoring_efficiency=None, regen_efficiency=None):
    profiles, samples = data["profiles"], data["samples"]
    motion = data["features"].reshape(profiles, samples, -1)
    target = data["target"].reshape(profiles, samples)
    base = data["base"].reshape(profiles, samples)
    sensitivity = data["sensitivity"].reshape(profiles, samples, -1)
    reference = data.get("reference_preload")
    if reference is not None:
        reference = reference.reshape(profiles, samples, -1)
    history = torch.zeros(
        (profiles, data["window_size"], 3), dtype=motion.dtype, device=motion.device
    )
    torque_rows, preload_rows = [], []
    torque_scale = max(float(data["torque_scale"]), 1e-9)
    for sample_index in range(samples):
        inputs = torch.cat((motion[:, sample_index, :], history.reshape(profiles, -1)), dim=1)
        preload = model(inputs)
        anchor = (
            model.neutral_preload
            if reference is None
            else reference[:, sample_index, :]
        )
        torque = base[:, sample_index] + torch.sum(
            sensitivity[:, sample_index, :] * (preload - anchor), dim=1
        )
        motor = target[:, sample_index] - torque
        realized = torch.stack((target[:, sample_index], torque, motor), dim=1) / torque_scale
        history = torch.cat((history[:, 1:, :], realized.detach().unsqueeze(1)), dim=1)
        torque_rows.append(torque)
        preload_rows.append(preload)
    return torch.stack(torque_rows, dim=1).reshape(-1), torch.stack(preload_rows, dim=1).reshape(-1, preload.shape[1])


def optimize_fixed_group_preload(data, group_assignment, group_limits, neutral, motoring_efficiency, regen_efficiency):
    """Coordinate-search a constant energy-minimizing starting preload."""
    device = data["target"].device
    assignment = torch.as_tensor(group_assignment, dtype=torch.long, device=device)
    values = torch.as_tensor(neutral, dtype=torch.float32, device=device)
    limits = torch.as_tensor(group_limits, dtype=torch.float32, device=device)

    def objective(group_values):
        spring_preload = group_values[assignment].unsqueeze(0).expand(len(data["target"]), -1)
        reference = values.new_tensor(neutral)[assignment]
        torque = data["base"] + torch.sum(data["sensitivity"] * (spring_preload - reference), dim=1)
        power = torch_energy_burden_power(
            (data["target"] - torque) * data["theta_dot"], motoring_efficiency, regen_efficiency
        )
        return torch.mean(power)

    with torch.no_grad():
        for _ in range(3):
            for group in range(len(values)):
                candidates = torch.linspace(0.0, float(limits[group]), 11, device=device)
                scores = []
                for candidate in candidates:
                    trial = values.clone()
                    trial[group] = candidate
                    scores.append(objective(trial))
                values[group] = candidates[int(torch.argmin(torch.stack(scores)))]
    return values.cpu().numpy()


def full_relaxed_preload_torque(
    dataset, preload, topology_path, device, batch_size, relaxation_steps, return_lengths=False,
    tension_only=False, cubic_ratio=0.0, cubic_reference_extension=0.05,
    return_sensitivity=False,
):
    """Evaluate commanded rest lengths with full nonlinear batched relaxation."""
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, device)
    fixed_k = torch.as_tensor(
        [spring.stiffness_k for spring in network.springs], dtype=torch.float32, device=device
    )
    cubic_k = fixed_k * cubic_ratio / max(cubic_reference_extension ** 2, 1e-12)
    theta_all = torch.as_tensor(dataset["theta"], dtype=torch.float32, device=device)
    preload_all = preload.to(device)
    outputs = []
    length_outputs = []
    sensitivity_outputs = []
    for start in range(0, len(theta_all), batch_size):
        stop = min(start + batch_size, len(theta_all))
        theta = theta_all[start:stop]
        commanded_rest = topology["rest_lengths"].unsqueeze(0) - preload_all[start:stop]
        prescribed = torch_prescribed_positions(topology, theta)
        internal = topology["internal_indices"]
        if internal.numel() and relaxation_steps:
            internal_positions = prescribed[:, internal, :].detach().clone().requires_grad_(True)
            relax_optimizer = torch.optim.Adam([internal_positions], lr=0.03)
            for _ in range(relaxation_steps):
                relax_optimizer.zero_grad(set_to_none=True)
                positions = prescribed.clone()
                positions[:, internal, :] = internal_positions
                delta = positions[:, topology["spring_b"], :] - positions[:, topology["spring_a"], :]
                length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
                stretch = length - commanded_rest
                if tension_only:
                    stretch = torch.clamp(stretch, min=0.0)
                energy = torch.sum(
                    0.5 * fixed_k.unsqueeze(0) * stretch ** 2
                    + 0.25 * cubic_k.unsqueeze(0) * stretch ** 4
                )
                energy.backward()
                relax_optimizer.step()
            positions = prescribed.clone()
            positions[:, internal, :] = internal_positions.detach()
        else:
            positions = prescribed

        a, b = topology["spring_a"], topology["spring_b"]
        delta = positions[:, b, :] - positions[:, a, :]
        length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
        direction = delta / length.unsqueeze(2)
        stretch = length - commanded_rest
        if tension_only:
            stretch = torch.clamp(stretch, min=0.0)
        force_on_a = fixed_k.unsqueeze(0) * stretch + cubic_k.unsqueeze(0) * stretch ** 3
        force_on_a = force_on_a.unsqueeze(2) * direction
        torque = torch.zeros(len(theta), dtype=torch.float32, device=device)
        sensitivity = torch.zeros(
            (len(theta), len(a)), dtype=torch.float32, device=device
        )
        tangent = fixed_k.unsqueeze(0) + 3.0 * cubic_k.unsqueeze(0) * stretch**2
        if tension_only:
            tangent = tangent * (stretch > 0.0)
        sensitivity_force_on_a = tangent.unsqueeze(2) * direction
        limb2 = set(int(i) for i in topology["limb2_indices"].detach().cpu().numpy())
        for spring_index in range(len(a)):
            node_a, node_b = int(a[spring_index]), int(b[spring_index])
            if node_a in limb2:
                r, force = positions[:, node_a, :], force_on_a[:, spring_index, :]
                torque += r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
                derivative_force = sensitivity_force_on_a[:, spring_index, :]
                sensitivity[:, spring_index] += (
                    r[:, 0] * derivative_force[:, 1] - r[:, 1] * derivative_force[:, 0]
                )
            if node_b in limb2:
                r, force = positions[:, node_b, :], -force_on_a[:, spring_index, :]
                torque += r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
                derivative_force = -sensitivity_force_on_a[:, spring_index, :]
                sensitivity[:, spring_index] += (
                    r[:, 0] * derivative_force[:, 1] - r[:, 1] * derivative_force[:, 0]
                )
        outputs.append(torque.detach())
        if return_lengths:
            length_outputs.append(length.detach())
        if return_sensitivity:
            sensitivity_outputs.append(sensitivity.detach())
    torque = torch.cat(outputs)
    extras = []
    if return_lengths:
        extras.append(torch.cat(length_outputs))
    if return_sensitivity:
        extras.append(torch.cat(sensitivity_outputs))
    if extras:
        return (torque, *extras)
    return torque


def causal_relaxed_preload_schedule(
    model,
    dataset,
    topology_path,
    device,
    batch_size,
    relaxation_steps,
    tension_only=False,
    cubic_ratio=0.0,
    cubic_reference_extension=0.05,
):
    """Generate commands using relaxed realized torque in the causal history."""
    data = tensors(dataset, device)
    profiles, samples = dataset["profiles"], dataset["samples"]
    motion = data["features"].reshape(profiles, samples, -1)
    target = data["target"].reshape(profiles, samples)
    theta = data["theta"].reshape(profiles, samples)
    history = torch.zeros(
        (profiles, dataset["window_size"], 3),
        dtype=motion.dtype,
        device=device,
    )
    torque_scale = max(float(dataset["torque_scale"]), 1e-9)
    schedule_rows = []
    for sample_index in range(samples):
        inputs = torch.cat(
            (motion[:, sample_index, :], history.reshape(profiles, -1)), dim=1
        )
        with torch.no_grad():
            preload = model(inputs)
        step_dataset = {
            "theta": theta[:, sample_index].detach().cpu().numpy(),
        }
        torque = full_relaxed_preload_torque(
            step_dataset,
            preload,
            topology_path,
            device,
            batch_size,
            relaxation_steps,
            tension_only=tension_only,
            cubic_ratio=cubic_ratio,
            cubic_reference_extension=cubic_reference_extension,
        )
        motor = target[:, sample_index] - torque
        realized = torch.stack(
            (target[:, sample_index], torque, motor), dim=1
        ) / torque_scale
        history = torch.cat((history[:, 1:, :], realized.unsqueeze(1)), dim=1)
        schedule_rows.append(preload)
    return torch.stack(schedule_rows, dim=1).reshape(-1, preload.shape[1])


def refresh_preload_surrogate(
    model,
    dataset,
    topology_path,
    device,
    batch_size,
    relaxation_steps,
    finite_difference,
    preload_limits,
    tension_only=False,
    cubic_ratio=0.0,
    cubic_reference_extension=0.05,
):
    """Linearize relaxed preload mechanics around the controller's schedule."""
    anchor_preload = causal_relaxed_preload_schedule(
        model,
        dataset,
        topology_path,
        device,
        batch_size,
        relaxation_steps,
        tension_only=tension_only,
        cubic_ratio=cubic_ratio,
        cubic_reference_extension=cubic_reference_extension,
    )
    anchor_torque, operating_length, sensitivity = full_relaxed_preload_torque(
        dataset,
        anchor_preload,
        topology_path,
        device,
        batch_size,
        relaxation_steps,
        return_lengths=True,
        return_sensitivity=True,
        tension_only=tension_only,
        cubic_ratio=cubic_ratio,
        cubic_reference_extension=cubic_reference_extension,
    )
    refreshed = dict(dataset)
    refreshed["base"] = anchor_torque.detach().cpu().numpy().astype(np.float32)
    refreshed["sensitivity"] = sensitivity.detach().cpu().numpy().astype(np.float32)
    refreshed["reference_preload"] = (
        anchor_preload.detach().cpu().numpy().astype(np.float32)
    )
    refreshed["operating_length"] = (
        operating_length.detach().cpu().numpy().astype(np.float32)
    )
    return refreshed


def preload_adjustment_energy(
    schedule,
    operating_length,
    nominal_rest,
    spring_k,
    cubic_k,
    tension_only=False,
):
    """Positive ideal work to change rest length at the current geometry."""
    previous_rest = nominal_rest - schedule[:, :-1, :]
    new_rest = nominal_rest - schedule[:, 1:, :]
    current_length = operating_length[:, 1:, :]
    before_stretch = current_length - previous_rest
    after_stretch = current_length - new_rest
    if tension_only:
        before_stretch = torch.clamp(before_stretch, min=0.0)
        after_stretch = torch.clamp(after_stretch, min=0.0)
    before = 0.5 * spring_k * before_stretch**2 + 0.25 * cubic_k * before_stretch**4
    after = 0.5 * spring_k * after_stretch**2 + 0.25 * cubic_k * after_stretch**4
    return torch.clamp(after - before, min=0.0)


def differentiable_relaxed_torque(
    theta, preload, topology_path, device, relaxation_steps=3, step_size=0.01,
    mechanics=None, tension_only=False,
):
    """Fully differentiable nonlinear geometry with unrolled internal-node relaxation."""
    if mechanics is None:
        network, _ = load_network(topology_path)
        topology = torch_topology_data(network, device)
        fixed_k = torch.as_tensor([s.stiffness_k for s in network.springs], dtype=torch.float32, device=device)
        limb2 = set(int(i) for i in topology["limb2_indices"].detach().cpu().numpy())
    else:
        topology, fixed_k, cubic_k, limb2 = mechanics
    if mechanics is None:
        cubic_k = torch.zeros_like(fixed_k)
    commanded_rest = topology["rest_lengths"].unsqueeze(0) - preload
    prescribed = torch_prescribed_positions(topology, theta)
    internal = topology["internal_indices"]
    internal_positions = prescribed[:, internal, :]
    for _ in range(relaxation_steps):
        internal_positions = internal_positions.requires_grad_(True)
        positions = prescribed.clone()
        positions[:, internal, :] = internal_positions
        delta = positions[:, topology["spring_b"], :] - positions[:, topology["spring_a"], :]
        length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
        stretch = length - commanded_rest
        if tension_only:
            stretch = torch.clamp(stretch, min=0.0)
        energy = torch.sum(0.5 * fixed_k * stretch ** 2 + 0.25 * cubic_k * stretch ** 4)
        gradient = torch.autograd.grad(energy, internal_positions, create_graph=True)[0]
        internal_positions = internal_positions - step_size * gradient
    positions = prescribed.clone()
    positions[:, internal, :] = internal_positions
    a, b = topology["spring_a"], topology["spring_b"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    stretch = length - commanded_rest
    if tension_only:
        stretch = torch.clamp(stretch, min=0.0)
    force_on_a = (fixed_k * stretch + cubic_k * stretch ** 3).unsqueeze(2) * delta / length.unsqueeze(2)
    torque = torch.zeros(len(theta), dtype=torch.float32, device=device)
    for spring_index in range(len(a)):
        node_a, node_b = int(a[spring_index]), int(b[spring_index])
        if node_a in limb2:
            r, force = positions[:, node_a, :], force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
        if node_b in limb2:
            r, force = positions[:, node_b, :], -force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
    return torque


def predict_nonlinear(
    model, data, topology_path, device, relaxation_steps, step_size,
    mechanics=None, tension_only=False,
):
    """Causal MLP rollout whose realized torque comes from full nonlinear mechanics."""
    profiles, samples = data["profiles"], data["samples"]
    motion = data["features"].reshape(profiles, samples, -1)
    theta = data["theta"].reshape(profiles, samples)
    target = data["target"].reshape(profiles, samples)
    history = torch.zeros((profiles, data["window_size"], 3), dtype=motion.dtype, device=device)
    torque_rows, preload_rows = [], []
    torque_scale = max(float(data["torque_scale"]), 1e-9)
    for sample_index in range(samples):
        inputs = torch.cat((motion[:, sample_index, :], history.reshape(profiles, -1)), dim=1)
        preload = model(inputs)
        torque = differentiable_relaxed_torque(
            theta[:, sample_index], preload, topology_path, device,
            relaxation_steps=relaxation_steps, step_size=step_size,
            mechanics=mechanics, tension_only=tension_only,
        )
        motor = target[:, sample_index] - torque
        realized = torch.stack((target[:, sample_index], torque, motor), dim=1) / torque_scale
        history = torch.cat((history[:, 1:, :], realized.detach().unsqueeze(1)), dim=1)
        torque_rows.append(torque)
        preload_rows.append(preload)
    return torch.stack(torque_rows, dim=1).reshape(-1), torch.stack(preload_rows, dim=1).reshape(-1, preload.shape[1])


def nonlinear_preload_schedule(
    model, data, topology_path, device, relaxation_steps, step_size, mechanics,
    batch_profiles, tension_only=False,
):
    """Generate causal commands using nonlinear realized-torque history in bounded memory."""
    schedules = []
    for start in range(0, data["profiles"], batch_profiles):
        stop = min(start + batch_profiles, data["profiles"])
        batch = profile_batch(data, torch.arange(start, stop, device=device))
        with torch.enable_grad():
            _, preload = predict_nonlinear(
                model, batch, topology_path, device, relaxation_steps, step_size,
                mechanics=mechanics, tension_only=tension_only,
            )
        schedules.append(preload.detach().reshape(stop - start, data["samples"], -1))
    return torch.cat(schedules, dim=0).reshape(-1, schedules[0].shape[-1])


def energy_ledger(
    dataset, torque, preload, lengths, topology_path, motoring_efficiency,
    regen_efficiency, tension_only=False, cubic_ratio=0.0,
    cubic_reference_extension=0.05,
):
    """Average per-profile motor and ideal preload-actuator energy ledger."""
    network, _ = load_network(topology_path)
    device = torque.device
    target = torch.as_tensor(dataset["target"], dtype=torch.float32, device=device)
    theta_dot = torch.as_tensor(dataset["theta_dot"], dtype=torch.float32, device=device)
    dt = dataset["duration"] / max(dataset["samples"] - 1, 1)
    profiles = dataset["profiles"]
    baseline_power = torch_energy_burden_power(
        target * theta_dot, motoring_efficiency, regen_efficiency
    )
    residual_power = torch_energy_burden_power(
        (target - torque) * theta_dot, motoring_efficiency, regen_efficiency
    )
    baseline_motor_j = torch.sum(baseline_power) * dt / profiles
    residual_motor_j = torch.sum(residual_power) * dt / profiles

    k = torch.as_tensor(
        [spring.stiffness_k for spring in network.springs], dtype=torch.float32, device=device
    )
    k3 = k * cubic_ratio / max(cubic_reference_extension ** 2, 1e-12)
    rest = torch.as_tensor(
        [spring.rest_length for spring in network.springs], dtype=torch.float32, device=device
    )
    p = preload.reshape(profiles, dataset["samples"], -1)
    length = lengths.reshape(profiles, dataset["samples"], -1)
    previous_rest = rest - p[:, :-1, :]
    new_rest = rest - p[:, 1:, :]
    current_length = length[:, 1:, :]
    before_stretch = current_length - previous_rest
    after_stretch = current_length - new_rest
    if tension_only:
        before_stretch = torch.clamp(before_stretch, min=0.0)
        after_stretch = torch.clamp(after_stretch, min=0.0)
    energy_before = 0.5 * k * before_stretch ** 2 + 0.25 * k3 * before_stretch ** 4
    energy_after = 0.5 * k * after_stretch ** 2 + 0.25 * k3 * after_stretch ** 4
    adjustment_work = energy_after - energy_before
    preload_used_j = torch.sum(torch.clamp(adjustment_work, min=0.0)) / profiles
    preload_released_j = torch.sum(torch.clamp(-adjustment_work, min=0.0)) / profiles
    motor_saved_j = baseline_motor_j - residual_motor_j
    return {
        "baseline_motor_energy_j": float(baseline_motor_j.cpu()),
        "residual_motor_energy_j": float(residual_motor_j.cpu()),
        "motor_energy_saved_j": float(motor_saved_j.cpu()),
        "preload_adjustment_energy_used_j": float(preload_used_j.cpu()),
        "preload_energy_released_j": float(preload_released_j.cpu()),
        "net_energy_saved_after_preload_j": float((motor_saved_j - preload_used_j).cpu()),
    }


def metrics(model, dataset, device, motoring_efficiency, regen_efficiency, torque_override=None, preload_override=None):
    data = tensors(dataset, device)
    with torch.no_grad():
        torque, preload = predict(model, data, motoring_efficiency, regen_efficiency)
        if torque_override is not None:
            torque = torque_override
        if preload_override is not None:
            preload = preload_override
    target = data["target"]
    rmse = float(torch.sqrt(torch.mean((torque - target) ** 2)).cpu())
    zero_rmse = float(torch.sqrt(torch.mean((data["base"] - target) ** 2)).cpu())
    schedule = preload.reshape(dataset["profiles"], dataset["samples"], -1)
    change = torch.abs(schedule[:, 1:, :] - schedule[:, :-1, :])
    baseline_burden = torch.mean(
        torch_energy_burden_power(
            target * data["theta_dot"], motoring_efficiency, regen_efficiency
        )
    )
    fixed_burden = torch.mean(
        torch_energy_burden_power(
            (target - data["base"]) * data["theta_dot"],
            motoring_efficiency,
            regen_efficiency,
        )
    )
    controlled_burden = torch.mean(
        torch_energy_burden_power(
            (target - torque) * data["theta_dot"], motoring_efficiency, regen_efficiency
        )
    )
    return {
        "rmse_nm": rmse,
        "fixed_preload_rmse_nm": zero_rmse,
        "rmse_improvement_pct": 100.0 * (zero_rmse - rmse) / max(zero_rmse, 1e-9),
        "mean_preload_mm": float(torch.mean(preload).cpu()) * 1000.0,
        "mean_abs_preload_change_mm": float(torch.mean(change).cpu()) * 1000.0,
        "max_abs_preload_change_mm": float(torch.max(change).cpu()) * 1000.0,
        "fixed_preload_offload_pct": float(
            (100.0 * (baseline_burden - fixed_burden) / torch.clamp(baseline_burden, min=1e-9)).cpu()
        ),
        "controlled_preload_offload_pct": float(
            (100.0 * (baseline_burden - controlled_burden) / torch.clamp(baseline_burden, min=1e-9)).cpu()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-per-family", type=int, default=2000)
    parser.add_argument("--test-profiles-per-family", type=int, default=400)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--motion-mode", choices=("randomized", "triangular"), default="randomized")
    parser.add_argument("--fixed-frequency-hz", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--objective",
        choices=["torque-mse", "net-energy", "hybrid"],
        default="torque-mse",
        help="Training objective. Net energy includes residual motor energy and positive preload work.",
    )
    parser.add_argument(
        "--torque-loss-weight",
        type=float,
        default=0.0,
        help="Weight on torque MSE normalized by the dataset torque scale squared.",
    )
    parser.add_argument("--motor-energy-weight", type=float, default=1.0)
    parser.add_argument("--preload-work-weight", type=float, default=1.0)
    parser.add_argument(
        "--surrogate-refreshes",
        type=int,
        default=0,
        help="Relax and rebuild the local preload torque/work surrogate this many times during training.",
    )
    parser.add_argument(
        "--fixed-preload-initialization", action=argparse.BooleanOptionalAction, default=False,
        help="Initialize at a surrogate-optimized fixed preload. Disabled by default to retain signed adaptive authority.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--max-preload-mm", type=float, default=10.0)
    parser.add_argument("--neutral-preload-mm", type=float, default=5.0)
    parser.add_argument("--preload-change-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.35)
    parser.add_argument(
        "--electrical-loss-weight",
        type=float,
        default=1.0,
        help="Weight on residual motor electrical-energy ratio in the training loss; use 0 to remove it.",
    )
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--group-mode", choices=["three", "four", "six", "eight", "per-spring"], default="per-spring")
    parser.add_argument(
        "--tension-only", action=argparse.BooleanOptionalAction, default=False,
        help="Use slack tension-only springs during final nonlinear evaluation and energy accounting.",
    )
    parser.add_argument(
        "--classification-mode",
        choices=["roughness", "periodicity-high", "periodicity-medium", "periodicity-low"],
        default="roughness",
    )
    parser.add_argument("--motoring-efficiency", type=float, default=DEFAULT_MOTORING_EFFICIENCY)
    parser.add_argument("--regen-efficiency", type=float, default=DEFAULT_REGEN_EFFICIENCY)
    parser.add_argument("--finite-difference-mm", type=float, default=1.0)
    parser.add_argument("--nonlinear-batch-size", type=int, default=4096)
    parser.add_argument("--nonlinear-relaxation-steps", type=int, default=30)
    parser.add_argument(
        "--cubic-ratio", type=float, default=0.0,
        help="Dimensionless hardening at the reference extension; 0 preserves linear springs.",
    )
    parser.add_argument("--cubic-reference-extension-mm", type=float, default=50.0)
    parser.add_argument(
        "--cubic-design-extension-mm", type=float, default=1000.0,
        help="Largest absolute extension over which positive tangent stiffness is guaranteed.",
    )
    parser.add_argument(
        "--cubic-min-tangent-ratio", type=float, default=0.05,
        help="Minimum allowed dF/dx divided by linear stiffness at the design extension.",
    )
    parser.add_argument(
        "--minimum-rest-length-mm", type=float, default=5.0,
        help="Never command a spring rest length below this value.",
    )
    parser.add_argument(
        "--full-nonlinear-train-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use full nonlinear relaxed mechanics for final training-set metrics as well as test metrics.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output-name", default="preload_20spring_test")
    parser.add_argument(
        "--topology",
        default=PROJECT_ROOT / "topologies" / "preload" / "preload_fan_soft_015_long150.json",
        help="Selected preload topology derived from the established 20-spring internal fan.",
    )
    args = parser.parse_args()
    if args.fixed_frequency_hz is not None and args.fixed_frequency_hz <= 0.0:
        parser.error("--fixed-frequency-hz must be positive")
    if args.surrogate_refreshes < 0 or args.surrogate_refreshes >= args.iterations:
        parser.error("--surrogate-refreshes must be nonnegative and smaller than --iterations")
    if min(args.torque_loss_weight, args.motor_energy_weight, args.preload_work_weight) < 0.0:
        parser.error("Objective weights cannot be negative")

    # A trailing PowerShell backslash turns the intended stem into a directory
    # (for example ``preload_large\\.pt``), which PyTorch cannot use as a zip
    # archive filename. Keep output_name strictly as a filename stem.
    sanitized_output_name = args.output_name.strip().rstrip("\\/")
    sanitized_output_name = Path(sanitized_output_name).name
    for suffix in (".pt", ".pth"):
        if sanitized_output_name.lower().endswith(suffix):
            sanitized_output_name = sanitized_output_name[: -len(suffix)]
    if not sanitized_output_name:
        raise ValueError("--output-name must contain a filename stem, not only slashes")
    args.output_name = sanitized_output_name
    validate_efficiencies(args.motoring_efficiency, args.regen_efficiency)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.classification_mode == "roughness":
        train_params = generate_classified_profile_parameters(rng, args.profiles_per_family)
        test_params = generate_classified_profile_parameters(rng, args.test_profiles_per_family)
    else:
        periodicity_class = args.classification_mode.removeprefix("periodicity-")
        train_params = generate_periodicity_profiles(
            rng, args.profiles_per_family, args.duration, args.samples, args.seed + 3000, periodicity_class
        )
        test_params = generate_periodicity_profiles(
            rng, args.test_profiles_per_family, args.duration, args.samples, args.seed + 4000, periodicity_class
        )
    topology = Path(args.topology)
    angles = np.linspace(-ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD, 61)
    base, sensitivity = preload_mechanics(
        topology,
        angles,
        args.finite_difference_mm / 1000.0,
        reference_preload=args.neutral_preload_mm / 1000.0,
    )
    scales = normalization_scales(
        train_params,
        args.duration,
        args.samples,
        args.seed + 5000,
        args.window_size,
        motion_mode=args.motion_mode,
        fixed_frequency_hz=args.fixed_frequency_hz,
    )
    train = build_dataset(
        train_params, args.duration, args.samples, args.seed + 1000, angles, base, sensitivity,
        args.window_size, scales, args.motion_mode, args.fixed_frequency_hz,
    )
    test = build_dataset(
        test_params, args.duration, args.samples, args.seed + 2000, angles, base, sensitivity,
        args.window_size, scales, args.motion_mode, args.fixed_frequency_hz,
    )

    network, _ = load_network(topology)
    nominal_rest = np.asarray([spring.rest_length for spring in network.springs], dtype=float)
    minimum_rest = args.minimum_rest_length_mm / 1000.0
    per_spring_preload_limit = np.minimum(
        args.max_preload_mm / 1000.0,
        np.maximum(nominal_rest - minimum_rest, 0.0),
    )
    per_spring_neutral = np.minimum(
        args.neutral_preload_mm / 1000.0, per_spring_preload_limit
    )
    group_assignment = preload_groups(network, args.group_mode)
    group_count = int(group_assignment.max()) + 1
    group_limits = np.asarray([
        np.min(per_spring_preload_limit[group_assignment == group]) for group in range(group_count)
    ])
    group_reference = np.asarray([
        np.min(per_spring_neutral[group_assignment == group]) for group in range(group_count)
    ])
    train_data = tensors(train, device)
    if args.fixed_preload_initialization:
        fixed_group_preload = optimize_fixed_group_preload(
            train_data,
            group_assignment,
            group_limits,
            group_reference,
            args.motoring_efficiency,
            args.regen_efficiency,
        )
        print("Fixed energy-optimal group preload [mm]:", np.round(1000.0 * fixed_group_preload, 3))
    else:
        fixed_group_preload = group_reference.copy()
        print("Neutral midpoint group preload [mm]:", np.round(1000.0 * fixed_group_preload, 3))
    model = PreloadController(
        train["features"].shape[1] + 3 * args.window_size,
        args.hidden_dim,
        group_assignment,
        group_limits,
        fixed_group_preload,
        group_reference,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    spring_k = torch.as_tensor(
        [spring.stiffness_k for spring in network.springs], dtype=torch.float32, device=device
    )
    cubic_reference = args.cubic_reference_extension_mm / 1000.0
    if cubic_reference <= 0.0:
        parser.error("--cubic-reference-extension-mm must be positive")
    design_extension = args.cubic_design_extension_mm / 1000.0
    if design_extension <= 0.0:
        parser.error("--cubic-design-extension-mm must be positive")
    tangent_ratio = 1.0 + 3.0 * args.cubic_ratio * (design_extension / cubic_reference) ** 2
    if tangent_ratio < args.cubic_min_tangent_ratio:
        parser.error(
            "Unsafe cubic softening: tangent stiffness would fall to "
            f"{tangent_ratio:.3f} times the linear stiffness at the design extension. "
            "Use a less-negative ratio or a larger reference extension."
        )
    cubic_k = spring_k * args.cubic_ratio / cubic_reference ** 2
    nominal_rest_tensor = torch.as_tensor(
        nominal_rest, dtype=torch.float32, device=device
    ).reshape(1, 1, -1)
    print(
        "Training mechanics: causal relaxed operating-point preload surrogate"
        if args.surrogate_refreshes or args.objective != "torque-mse"
        else "Training mechanics: causal finite-difference preload torque surrogate"
    )
    print(f"Training objective: {args.objective}")
    active_train = train
    if args.surrogate_refreshes or args.objective != "torque-mse":
        print("Building initial relaxed preload torque/work surrogate...")
        active_train = refresh_preload_surrogate(
            model,
            active_train,
            topology,
            device,
            args.nonlinear_batch_size,
            args.nonlinear_relaxation_steps,
            args.finite_difference_mm / 1000.0,
            per_spring_preload_limit,
            tension_only=args.tension_only,
            cubic_ratio=args.cubic_ratio,
            cubic_reference_extension=cubic_reference,
        )
        train_data = tensors(active_train, device)
    phase_count = args.surrogate_refreshes + 1
    phase_iterations = [
        args.iterations // phase_count
        + (1 if index < args.iterations % phase_count else 0)
        for index in range(phase_count)
    ]
    refresh_after = set(np.cumsum(phase_iterations)[:-1].tolist())
    history = []
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        torque, preload = predict(model, train_data, args.motoring_efficiency, args.regen_efficiency)
        mse = torch.mean((torque - train_data["target"]) ** 2)
        schedule = preload.reshape(train["profiles"], train["samples"], -1)
        baseline_power = torch_energy_burden_power(
            train_data["target"] * train_data["theta_dot"],
            args.motoring_efficiency,
            args.regen_efficiency,
        )
        assisted_power = torch_energy_burden_power(
            (train_data["target"] - torque) * train_data["theta_dot"],
            args.motoring_efficiency,
            args.regen_efficiency,
        )
        energy_ratio = torch.mean(assisted_power) / torch.clamp(torch.mean(baseline_power), min=1e-9)
        electrical_term = torch.zeros_like(energy_ratio)

        # Match the final ledger: change rest length at the current relaxed
        # geometry, charge positive spring-energy changes, and recover none.
        if "operating_length" in train_data:
            operating_length = train_data["operating_length"].reshape(
                train["profiles"], train["samples"], -1
            )
            adjustment_work = preload_adjustment_energy(
                schedule,
                operating_length,
                nominal_rest_tensor,
                spring_k,
                cubic_k,
                tension_only=args.tension_only,
            )
        else:
            # Legacy torque-MSE runs do not use this telemetry for gradients.
            adjustment_work = torch.clamp(
                0.5 * spring_k * schedule[:, 1:, :] ** 2
                - 0.5 * spring_k * schedule[:, :-1, :] ** 2,
                min=0.0,
            )
        baseline_energy = torch.mean(baseline_power) * train["duration"]
        adjustment_ratio = torch.mean(torch.sum(adjustment_work, dim=(1, 2))) / torch.clamp(
            baseline_energy, min=1e-9
        )
        torque_term = mse / max(train["torque_scale"] ** 2, 1e-9)
        normalized_delta = (schedule[:, 1:, :] - schedule[:, :-1, :]) / torch.clamp(
            model.max_preload[model.group_assignment].reshape(1, 1, -1), min=1e-9
        )
        smoothness_term = torch.mean(normalized_delta**2)
        electrical_term = args.motor_energy_weight * energy_ratio
        preload_work_term = args.preload_work_weight * adjustment_ratio
        if args.objective == "torque-mse":
            loss = mse
        elif args.objective == "net-energy":
            loss = electrical_term + preload_work_term
        else:
            loss = (
                electrical_term
                + preload_work_term
                + args.torque_loss_weight * torque_term
                + args.smoothness_weight * smoothness_term
            )
        loss.backward()
        optimizer.step()
        gross_motor_offload_pct = 100.0 * (1.0 - energy_ratio)
        preload_cost_pct = 100.0 * adjustment_ratio
        net_system_offload_pct = gross_motor_offload_pct - preload_cost_pct
        gross_motor_saved_j = baseline_energy * (1.0 - energy_ratio)
        preload_adjustment_j = baseline_energy * adjustment_ratio
        net_system_saved_j = gross_motor_saved_j - preload_adjustment_j
        history.append({
            "iteration": iteration,
            "rmse_nm": float(torch.sqrt(mse).detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "motor_energy_ratio": float(energy_ratio.detach().cpu()),
            "electrical_term": float(electrical_term.detach().cpu()),
            "preload_work_ratio": float(adjustment_ratio.detach().cpu()),
            "preload_work_term": float(preload_work_term.detach().cpu()),
            "torque_term": float(torque_term.detach().cpu()),
            "smoothness_term": float(smoothness_term.detach().cpu()),
            "gross_motor_offload_pct": float(gross_motor_offload_pct.detach().cpu()),
            "preload_cost_pct": float(preload_cost_pct.detach().cpu()),
            "net_system_offload_pct": float(net_system_offload_pct.detach().cpu()),
            "gross_motor_saved_j": float(gross_motor_saved_j.detach().cpu()),
            "preload_adjustment_j": float(preload_adjustment_j.detach().cpu()),
            "net_system_saved_j": float(net_system_saved_j.detach().cpu()),
        })
        if iteration == 1 or iteration == args.iterations or iteration % 100 == 0:
            latest = history[-1]
            print(
                f"iteration {iteration:5d} | RMSE {latest['rmse_nm']:8.3f} Nm | "
                f"loss {latest['loss']:.5f} | gross motor offload {latest['gross_motor_offload_pct']:7.2f}% | "
                f"preload cost {latest['preload_cost_pct']:7.2f}% | "
                f"net offload {latest['net_system_offload_pct']:7.2f}% | "
                f"net saved {latest['net_system_saved_j']:8.3f} J"
            )
        if iteration in refresh_after:
            refresh_number = sorted(refresh_after).index(iteration) + 1
            print(
                f"Refreshing relaxed preload surrogate "
                f"({refresh_number}/{args.surrogate_refreshes})..."
            )
            active_train = refresh_preload_surrogate(
                model,
                active_train,
                topology,
                device,
                args.nonlinear_batch_size,
                args.nonlinear_relaxation_steps,
                args.finite_difference_mm / 1000.0,
                per_spring_preload_limit,
                tension_only=args.tension_only,
                cubic_ratio=args.cubic_ratio,
                cubic_reference_extension=cubic_reference,
            )
            train_data = tensors(active_train, device)

    train_data = tensors(train, device)
    if args.full_nonlinear_train_eval:
        train_preload = causal_relaxed_preload_schedule(
            model, train, topology, device, args.nonlinear_batch_size,
            args.nonlinear_relaxation_steps, tension_only=args.tension_only,
            cubic_ratio=args.cubic_ratio, cubic_reference_extension=cubic_reference,
        )
        nonlinear_train_torque, nonlinear_train_lengths = full_relaxed_preload_torque(
            train,
            train_preload,
            topology,
            device,
            args.nonlinear_batch_size,
            args.nonlinear_relaxation_steps,
            return_lengths=True,
            tension_only=args.tension_only,
            cubic_ratio=args.cubic_ratio,
            cubic_reference_extension=cubic_reference,
        )
        train_metrics = metrics(
            model, train, device, args.motoring_efficiency, args.regen_efficiency,
            torque_override=nonlinear_train_torque, preload_override=train_preload,
        )
        train_energy = energy_ledger(
            train, nonlinear_train_torque, train_preload, nonlinear_train_lengths, topology,
            args.motoring_efficiency, args.regen_efficiency,
            tension_only=args.tension_only,
            cubic_ratio=args.cubic_ratio,
            cubic_reference_extension=cubic_reference,
        )
    else:
        with torch.no_grad():
            _, train_preload = predict(
                model, train_data, args.motoring_efficiency, args.regen_efficiency
            )
        train_metrics = metrics(
            model, train, device, args.motoring_efficiency, args.regen_efficiency
        )
        train_energy = None
    test_data = tensors(test, device)
    test_preload = causal_relaxed_preload_schedule(
        model, test, topology, device, args.nonlinear_batch_size,
        args.nonlinear_relaxation_steps, tension_only=args.tension_only,
        cubic_ratio=args.cubic_ratio, cubic_reference_extension=cubic_reference,
    )
    nonlinear_test_torque, nonlinear_test_lengths = full_relaxed_preload_torque(
        test,
        test_preload,
        topology,
        device,
        args.nonlinear_batch_size,
        args.nonlinear_relaxation_steps,
        return_lengths=True,
        tension_only=args.tension_only,
        cubic_ratio=args.cubic_ratio,
        cubic_reference_extension=cubic_reference,
    )
    test_metrics = metrics(
        model,
        test,
        device,
        args.motoring_efficiency,
        args.regen_efficiency,
        torque_override=nonlinear_test_torque,
        preload_override=test_preload,
    )
    test_energy = energy_ledger(
        test, nonlinear_test_torque, test_preload, nonlinear_test_lengths, topology,
        args.motoring_efficiency, args.regen_efficiency,
        tension_only=args.tension_only,
        cubic_ratio=args.cubic_ratio,
        cubic_reference_extension=cubic_reference,
    )
    train_metrics.update(train_energy or {})
    test_metrics.update(test_energy)
    output_model = PROJECT_ROOT / "models" / "preload" / f"{args.output_name}.pt"
    output_table = PROJECT_ROOT / "tables" / "preload" / f"{args.output_name}_metrics.csv"
    output_plot = PROJECT_ROOT / "plots" / "preload" / "dataset_examples" / f"{args.output_name}_convergence.png"
    output_schedule_plot = PROJECT_ROOT / "plots" / "preload" / "dataset_examples" / f"{args.output_name}_schedule.png"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "history": history}, output_model)
    with output_table.open("w", newline="", encoding="utf-8") as handle:
        metric_fields = list(dict.fromkeys([*train_metrics, *test_metrics]))
        writer = csv.DictWriter(handle, fieldnames=["split", *metric_fields])
        writer.writeheader()
        writer.writerow({"split": "train", **train_metrics})
        writer.writerow({"split": "test", **test_metrics})
    iterations = [x["iteration"] for x in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(iterations, [x["loss"] for x in history])
    axes[0, 0].set_ylabel("Actual training objective")
    axes[0, 0].set_title("Total loss")
    axes[0, 1].plot(iterations, [x["gross_motor_offload_pct"] for x in history], label="gross joint-motor offload")
    axes[0, 1].plot(iterations, [x["preload_cost_pct"] for x in history], label="preload actuator cost")
    axes[0, 1].plot(iterations, [x["net_system_offload_pct"] for x in history], label="net offload")
    axes[0, 1].set_ylabel("Percent of baseline energy [%]")
    axes[0, 1].set_title("Training energy accounting (preload surrogate)")
    axes[0, 1].legend()
    axes[1, 0].plot(iterations, [x["rmse_nm"] for x in history])
    axes[1, 0].set_ylabel("Surrogate RMSE [Nm]")
    axes[1, 0].set_title("Surrogate torque fit (training objective)")
    axes[1, 1].plot(iterations, [x["electrical_term"] for x in history], label="motor electrical term")
    axes[1, 1].plot(iterations, [x["preload_work_ratio"] for x in history], label="preload work term")
    axes[1, 1].plot(iterations, [x["torque_term"] for x in history], label="torque term")
    axes[1, 1].plot(iterations, [x["smoothness_term"] for x in history], label="smoothness term")
    axes[1, 1].set_ylabel("Loss contribution")
    axes[1, 1].set_title("Loss components")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
    fig.savefig(output_plot, dpi=160)
    plt.close(fig)

    # Held-out causal behavior for one complete profile.
    count = test["samples"]
    time = np.linspace(0.0, test["duration"], count)
    schedule = test_preload[:count].detach().cpu().numpy()
    group_schedule = np.column_stack([
        np.mean(schedule[:, group_assignment == group], axis=1) * 1000.0
        for group in range(group_count)
    ])
    target_np = test_data["target"][:count].detach().cpu().numpy()
    spring_np = nonlinear_test_torque[:count].detach().cpu().numpy()
    residual_np = target_np - spring_np
    combined_np = spring_np + residual_np
    theta_np = test["theta"][:count]
    velocity_np = test["theta_dot"][:count]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    group_names = tuple(f"group {index + 1}" for index in range(group_count))
    for group in range(group_count):
        axes[0].plot(time, group_schedule[:, group], label=group_names[group])
    axes[0].set_ylabel("Preload [mm]"); axes[0].legend(ncol=2); axes[0].set_title("Adaptive preload commands")
    axes[1].plot(time, theta_np, label="angle [rad]")
    axes[1].plot(time, velocity_np / 10.0, label="velocity / 10")
    axes[1].set_ylabel("Motion"); axes[1].legend()
    axes[2].plot(time, combined_np, color="tab:green", linestyle=":", linewidth=3.0, label="spring + motor", zorder=1)
    axes[2].plot(time, spring_np, color="tab:blue", linewidth=2.5, label="spring torque", zorder=3)
    axes[2].plot(time, residual_np, color="tab:red", linestyle="-.", linewidth=2.2, label="residual motor torque", zorder=3)
    axes[2].plot(time, target_np, color="black", linestyle="--", linewidth=2.0, label="demanded torque", zorder=4)
    axes[2].set_ylabel("Torque [Nm]"); axes[2].legend(ncol=2)
    axes[2].set_xlabel("Time [s]")
    angle_order = np.argsort(theta_np)
    angle_deg = np.rad2deg(theta_np)
    axes[3].scatter(angle_deg, spring_np, s=18, color="tab:blue", alpha=0.8, label="spring torque", zorder=3)
    axes[3].scatter(angle_deg, residual_np, s=16, color="tab:red", marker="x", alpha=0.75, label="residual motor torque", zorder=3)
    axes[3].plot(angle_deg[angle_order], target_np[angle_order], color="black", linestyle="--", linewidth=2.0, label="demanded torque", zorder=4)
    axes[3].set_xlabel("Joint angle [deg]")
    axes[3].set_ylabel("Torque [Nm]")
    axes[3].set_title("Torque-angle profile")
    axes[3].legend(ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.savefig(output_schedule_plot, dpi=160)
    plt.close(fig)

    print("\nPreload neural-network results")
    for split, values in (("train", train_metrics), ("test", test_metrics)):
        print(
            f"{split:5s} | fixed RMSE {values['fixed_preload_rmse_nm']:.3f} Nm | "
            f"controlled RMSE {values['rmse_nm']:.3f} Nm | improvement {values['rmse_improvement_pct']:.2f}% | "
            f"offload {values['controlled_preload_offload_pct']:.2f}% | "
            f"mean |dp| {values['mean_abs_preload_change_mm']:.3f} mm/step"
        )
        if "baseline_motor_energy_j" in values:
            print(
                f"      motor baseline {values['baseline_motor_energy_j']:.3f} J | "
                f"motor residual {values['residual_motor_energy_j']:.3f} J | "
                f"motor saved {values['motor_energy_saved_j']:.3f} J | "
                f"preload used {values['preload_adjustment_energy_used_j']:.3f} J | "
                f"net saved {values['net_energy_saved_after_preload_j']:.3f} J"
            )
    print(f"Saved model: {output_model}")
    print(f"Saved metrics: {output_table}")
    print(f"Saved plot: {output_plot}")
    print(f"Saved schedule: {output_schedule_plot}")
    print(f"{group_count} independent preload commands update every timestep around the neutral operating point.")
    if args.objective == "torque-mse":
        print("Training loss is torque MSE only.")
    elif args.objective == "net-energy":
        print("Training minimizes residual motor electrical energy plus positive preload-adjustment work.")
    else:
        print("Training uses net energy plus normalized torque and smoothness terms.")
    print("Inputs are causal motion and realized torque windows; no target-profile descriptor is used.")
    print("The final net-energy ledger charges positive ideal preload-adjustment work with zero recovery.")
    if args.objective == "torque-mse":
        print("Motor and preload energy are telemetry only and do not affect backpropagation.")
    else:
        print("Motor and preload energy both affect backpropagation.")
    print("Final nonlinear mechanics use tension-only slack springs." if args.tension_only else "Springs support tension and compression.")


if __name__ == "__main__":
    main()
