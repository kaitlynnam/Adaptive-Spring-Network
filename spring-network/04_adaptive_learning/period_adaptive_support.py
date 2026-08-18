"""Shared data and mechanics helpers for the causal period-buffer pipeline."""

from pathlib import Path

import numpy as np
import torch

from mechanics_3d import torque_components
from profile_generator import ANGLE_LIMIT_RAD, profile_torque


MAIN_MODEL_NAME = "period_adaptive_3d_60spring_bounded_extended"
MAIN_ARTIFACT_NAMES = {MAIN_MODEL_NAME, f"{MAIN_MODEL_NAME}_many_profiles"}


def figure_path(output_dir, model_name, filename):
    """Use concise names for the main model and prefixed names for experiments."""
    name = filename if model_name in MAIN_ARTIFACT_NAMES else f"{model_name}_{filename}"
    return Path(output_dir) / name


def causal_derivative(values, time):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    derivative = np.zeros_like(values)
    if len(values) > 1:
        derivative[1:] = np.diff(values) / np.diff(time)
    return derivative


def generate_motion_trajectory(params, duration, samples, seed,
                               motion_mode="triangular", fixed_frequency_hz=None):
    """Generate the prescribed motion and target torque for one period."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, duration, samples)
    frequency = params["frequency_hz"] if fixed_frequency_hz is None else float(fixed_frequency_hz)
    if motion_mode == "triangular":
        phase = np.mod(frequency * time, 1.0)
        theta = ANGLE_LIMIT_RAD * (1.0 - 4.0 * np.abs(phase - 0.5))
    elif motion_mode == "randomized":
        amplitude = np.deg2rad(params["amplitude_deg"])
        phase = params["phase"]
        theta = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
        theta += (params["harmonic_fraction"] * 0.18 * amplitude
                  * np.sin(np.pi * frequency * time + 0.4 * phase))
        for _ in range(params["bump_count"]):
            center = rng.uniform(time[0] + 0.1 * duration, time[-1] - 0.1 * duration)
            width = rng.uniform(0.035, 0.14)
            height = rng.uniform(-0.18 * amplitude, 0.18 * amplitude)
            theta += height * np.exp(-0.5 * ((time - center) / width) ** 2)
        raw_noise = rng.normal(0.0, params["noise_scale"], size=samples)
        kernel_size = min(13, samples if samples % 2 else samples - 1)
        kernel_x = np.linspace(-2.5, 2.5, max(kernel_size, 1))
        kernel = np.exp(-0.5 * kernel_x**2)
        theta += np.convolve(raw_noise, kernel / np.sum(kernel), mode="same")
    else:
        raise ValueError("motion_mode must be 'triangular' or 'randomized'")
    theta = np.clip(theta, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD)
    theta_dot = causal_derivative(theta, time)
    theta_ddot = causal_derivative(theta_dot, time)
    return time, theta, theta_dot, theta_ddot, profile_torque(theta, params)


def interpolate_basis(basis_by_angle, angles_rad, theta):
    """Interpolate each spring's relaxed torque basis onto a trajectory."""
    return np.column_stack([
        np.interp(theta, angles_rad, basis_by_angle[:, spring_index],
                  left=basis_by_angle[0, spring_index],
                  right=basis_by_angle[-1, spring_index])
        for spring_index in range(basis_by_angle.shape[1])
    ])


def spatial_initial_basis(topology, angles, relaxation_steps):
    """Build the initial per-spring 3D torque basis."""
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(angles), 1)
    components = torque_components(
        topology,
        torch.as_tensor(angles, dtype=torch.float32, device=stiffness.device),
        stiffness,
        relaxation_steps,
    )
    return (components / torch.clamp(stiffness, min=1e-9)).detach().cpu().numpy()


def initialize_period_model(rng, input_dim, hidden_dim, output_dim, initial_k, min_k):
    """Initialize an MLP whose positive outputs begin near topology stiffness."""
    shifted = np.maximum(np.asarray(initial_k) - min_k, 1e-6)
    output_bias = shifted.copy()
    small = shifted <= 20.0
    output_bias[small] = np.log(np.expm1(shifted[small]))
    return {
        "w1": rng.normal(0.0, 0.15, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0.0, 0.02, size=(hidden_dim, output_dim)),
        "b2": output_bias,
    }
