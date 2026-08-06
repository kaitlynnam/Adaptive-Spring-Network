"""Shared trajectory and relaxed-mechanics utilities for passive controllers."""

from pathlib import Path
import sys

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from profile_generator import ANGLE_LIMIT_RAD, profile_torque  # noqa: E402
from topology_loader import load_network  # noqa: E402


PASSIVE_NETWORK_PRESETS = {
    "baseline": {
        "topology": PROJECT_ROOT / "topologies" / "profile_conditioned_passive" / "baseline_model.json",
    },
    "fan": {
        "topology": PROJECT_ROOT / "topologies" / "profile_conditioned_passive" / "internal_fan_20_spring_model.json",
    },
}


def causal_derivative(values, t):
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


def _smooth_noise(rng, samples, scale):
    raw = rng.normal(0.0, scale, size=samples)
    kernel_size = min(13, samples if samples % 2 == 1 else samples - 1)
    kernel_size = max(kernel_size, 1)
    x = np.linspace(-2.5, 2.5, kernel_size)
    kernel = np.exp(-0.5 * x**2)
    kernel /= np.sum(kernel)
    return np.convolve(raw, kernel, mode="same")


def _add_irregular_bumps(rng, t, theta, count, max_height):
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
    """Generate motion samples used to evaluate one passive profile."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, samples)
    freq = params["frequency_hz"] if fixed_frequency_hz is None else float(fixed_frequency_hz)
    if motion_mode == "triangular":
        cycle_fraction = np.mod(freq * t, 1.0)
        theta = ANGLE_LIMIT_RAD * (1.0 - 4.0 * np.abs(cycle_fraction - 0.5))
    elif motion_mode == "randomized":
        amp = np.deg2rad(params["amplitude_deg"])
        phase = params["phase"]
        base = amp * np.sin(2.0 * np.pi * freq * t + phase)
        harmonic = 0.18 * amp * np.sin(2.0 * np.pi * 0.5 * freq * t + 0.4 * phase)
        theta = base + params["harmonic_fraction"] * harmonic
        theta = _add_irregular_bumps(rng, t, theta, params["bump_count"], 0.18 * amp)
        theta += _smooth_noise(rng, samples, params["noise_scale"])
    else:
        raise ValueError("motion_mode must be 'randomized' or 'triangular'")
    theta = np.clip(theta, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD)
    theta_dot = causal_derivative(theta, t)
    theta_ddot = causal_derivative(theta_dot, t)
    return t, theta, theta_dot, theta_ddot, profile_torque(theta, params)


def select_training_device(requested):
    if requested == "cpu":
        return "cpu"
    if torch is None:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested, but PyTorch is not installed.")
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return "cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu"


def interpolate_basis(basis_by_angle, angles_rad, theta):
    basis = np.empty((len(theta), basis_by_angle.shape[1]), dtype=float)
    for spring_index in range(basis_by_angle.shape[1]):
        basis[:, spring_index] = np.interp(
            theta, angles_rad, basis_by_angle[:, spring_index],
            left=basis_by_angle[0, spring_index], right=basis_by_angle[-1, spring_index],
        )
    return basis


def torch_topology_data(network, device):
    node_names = list(network.nodes)
    node_index = {name: index for index, name in enumerate(node_names)}
    node_types = [network.nodes[name].type for name in node_names]
    local_positions = np.vstack([network.nodes[name].local_position for name in node_names])
    return {
        "node_types": node_types,
        "local_positions": torch.as_tensor(local_positions, dtype=torch.float32, device=device),
        "spring_a": torch.as_tensor([node_index[s.node_a] for s in network.springs], dtype=torch.long, device=device),
        "spring_b": torch.as_tensor([node_index[s.node_b] for s in network.springs], dtype=torch.long, device=device),
        "rest_lengths": torch.as_tensor([s.rest_length for s in network.springs], dtype=torch.float32, device=device),
        "internal_indices": torch.as_tensor([i for i, kind in enumerate(node_types) if kind == "internal"], dtype=torch.long, device=device),
        "limb2_indices": torch.as_tensor([i for i, kind in enumerate(node_types) if kind == "limb2"], dtype=torch.long, device=device),
    }


def _prescribed_positions(topology, theta):
    local = topology["local_positions"]
    positions = local.unsqueeze(0).repeat(len(theta), 1, 1)
    c, s = torch.cos(theta), torch.sin(theta)
    for index, node_type in enumerate(topology["node_types"]):
        if node_type == "limb2":
            x, y = local[index, 0], local[index, 1]
            positions[:, index, 0] = c * x - s * y
            positions[:, index, 1] = s * x + c * y
    return positions


def _spring_energy(topology, positions, stiffness):
    delta = positions[:, topology["spring_b"], :] - positions[:, topology["spring_a"], :]
    stretch = torch.linalg.norm(delta, dim=2).clamp_min(1e-9) - topology["rest_lengths"].unsqueeze(0)
    return torch.sum(0.5 * stiffness * stretch**2)


def _relax_positions(topology, prescribed, stiffness, steps, learning_rate):
    internal_indices = topology["internal_indices"]
    if internal_indices.numel() == 0 or not steps:
        return prescribed
    internal = prescribed[:, internal_indices, :].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([internal], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        positions = prescribed.clone()
        positions[:, internal_indices, :] = internal
        _spring_energy(topology, positions, stiffness).backward()
        optimizer.step()
    positions = prescribed.clone()
    positions[:, internal_indices, :] = internal.detach()
    return positions


def torch_torque_components_batch(topology, theta, stiffness, relax_internal, relaxation_steps, relaxation_lr):
    positions = _prescribed_positions(topology, theta)
    if relax_internal:
        positions = _relax_positions(topology, positions, stiffness, relaxation_steps, relaxation_lr)
    a, b = topology["spring_a"], topology["spring_b"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    force_on_a = (stiffness * (length - topology["rest_lengths"].unsqueeze(0))).unsqueeze(2) * (delta / length.unsqueeze(2))
    components = torch.zeros((len(theta), len(a)), dtype=positions.dtype, device=positions.device)
    limb2 = set(int(index) for index in topology["limb2_indices"].detach().cpu().numpy())
    for spring_index in range(len(a)):
        node_a, node_b = int(a[spring_index]), int(b[spring_index])
        if node_a in limb2:
            r, force = positions[:, node_a, :], force_on_a[:, spring_index, :]
            components[:, spring_index] += r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
        if node_b in limb2:
            r, force = positions[:, node_b, :], -force_on_a[:, spring_index, :]
            components[:, spring_index] += r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
    return components


def torch_torque_from_dataset(dataset, topology_path, stiffness_schedule, relax_internal=True, device="auto", batch_size=8192, relaxation_steps=80, relaxation_lr=0.03, **_):
    selected = torch.device(select_training_device(device))
    network, _ = load_network(topology_path)
    topology = torch_topology_data(network, selected)
    theta = np.asarray(dataset["theta"], dtype=float)
    stiffness_schedule = np.asarray(stiffness_schedule, dtype=float)
    predicted = np.empty(len(theta), dtype=float)
    for start in range(0, len(theta), batch_size):
        stop = min(start + batch_size, len(theta))
        values = torch_torque_components_batch(
            topology,
            torch.as_tensor(theta[start:stop], dtype=torch.float32, device=selected),
            torch.as_tensor(stiffness_schedule[start:stop], dtype=torch.float32, device=selected),
            relax_internal, relaxation_steps, relaxation_lr,
        ).sum(dim=1)
        predicted[start:stop] = values.detach().cpu().numpy()
    return predicted


def differentiable_relaxed_stiffness_torque(topology, theta, stiffness, relaxation_steps, step_size, max_step):
    prescribed = _prescribed_positions(topology, theta)
    indices = topology["internal_indices"]
    internal = prescribed[:, indices, :]
    for _ in range(relaxation_steps):
        internal = internal.requires_grad_(True)
        positions = prescribed.clone()
        positions[:, indices, :] = internal
        gradient = torch.autograd.grad(_spring_energy(topology, positions, stiffness), internal, create_graph=True)[0]
        internal = internal + max_step * torch.tanh(-step_size * gradient / max(max_step, 1e-12))
    positions = prescribed.clone()
    positions[:, indices, :] = internal
    a, b = topology["spring_a"], topology["spring_b"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    force_on_a = (stiffness * (length - topology["rest_lengths"].unsqueeze(0))).unsqueeze(2) * (delta / length.unsqueeze(2))
    torque = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)
    limb2 = set(int(index) for index in topology["limb2_indices"].detach().cpu().numpy())
    for spring_index in range(len(a)):
        node_a, node_b = int(a[spring_index]), int(b[spring_index])
        if node_a in limb2:
            r, force = positions[:, node_a, :], force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
        if node_b in limb2:
            r, force = positions[:, node_b, :], -force_on_a[:, spring_index, :]
            torque = torque + r[:, 0] * force[:, 1] - r[:, 1] * force[:, 0]
    return torque
