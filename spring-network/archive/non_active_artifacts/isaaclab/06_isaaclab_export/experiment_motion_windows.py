"""Quick controlled motion/torque-history ablation on one terrain family."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "01_core_model"), str(PROJECT_ROOT / "04_adaptive_learning")]
from adaptive_model import ANGLE_DEGREES, initial_stiffnesses, save_model, spring_torque_basis
from topology_loader import load_network
from train_adaptive_dataset import interpolate_basis

ROLLOUT = PROJECT_ROOT / "data" / "isaaclab_rollouts" / "go2_rough_500env_labeled.npz"
FAMILY = "hf_pyramid_slope"
JOINT = "FL_calf_joint"
ITERATIONS = 100

CONFIGS = [
    ("motion_only_w1", 1, True, "none"),
    ("motion_only_w2", 2, True, "none"),
    ("motion_only_w3", 3, True, "none"),
    ("motion_only_w5", 5, True, "none"),
    ("motion_only_w8", 8, True, "none"),
    ("motion_only_w10", 10, True, "none"),
    ("motion_only_w15", 15, True, "none"),
    ("motion_only_w20", 20, True, "none"),
]


def windows(values, size):
    rows = []
    for index in range(len(values)):
        part = values[max(0, index - size + 1) : index + 1]
        if len(part) < size:
            part = np.vstack((np.repeat(part[:1], size - len(part), axis=0), part))
        rows.append(part.reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def motion_features(theta, velocity, t, env_ids, size, use_acceleration, scales):
    rows = []
    for env_id in env_ids:
        acceleration = np.zeros(len(t), dtype=float)
        acceleration[1:] = np.diff(velocity[:, env_id]) / np.diff(t[:, env_id])
        channels = [theta[:, env_id] / scales["theta"], velocity[:, env_id] / scales["velocity"]]
        if use_acceleration:
            channels.append(acceleration / scales["acceleration"])
        rows.append(windows(np.column_stack(channels), size))
    return np.stack(rows)


def run_model(train_motion, test_motion, train_basis, test_basis, train_target, test_target, history_mode, window, initial_k,
              iterations=ITERATIONS, return_model=False):
    device = torch.device("cuda")
    tensors = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in
               (train_motion, test_motion, train_basis, test_basis, train_target, test_target)]
    train_motion, test_motion, train_basis, test_basis, train_target, test_target = tensors
    history_channels = {"none": 0, "target": 1, "all": 3}[history_mode]
    input_dim = train_motion.shape[2] + window * history_channels
    generator = torch.Generator(device=device).manual_seed(42)
    w1 = torch.randn(input_dim, 128, generator=generator, device=device) * 0.1
    b1 = torch.zeros(128, device=device)
    w2 = torch.randn(128, train_basis.shape[2], generator=generator, device=device) * 0.02
    scaled = np.clip((initial_k - 1.0) / 799.0, 1e-6, 1 - 1e-6)
    b2 = torch.as_tensor(np.log(scaled / (1 - scaled)), dtype=torch.float32, device=device)
    parameters = [torch.nn.Parameter(x) for x in (w1, b1, w2, b2)]
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    initial = torch.as_tensor(initial_k, dtype=torch.float32, device=device)

    def rollout(motion, basis, target):
        count, samples = target.shape
        if history_channels == 0:
            flat_motion = motion.reshape(count * samples, -1)
            flat_basis = basis.reshape(count * samples, -1)
            hidden = torch.tanh(flat_motion @ parameters[0] + parameters[1])
            stiffness = 1.0 + 799.0 * torch.sigmoid(torch.clamp(hidden @ parameters[2] + parameters[3], -50, 50))
            prediction = torch.sum(flat_basis * stiffness, dim=1).reshape(count, samples)
            return prediction, stiffness
        history = torch.zeros(count, window, history_channels, device=device)
        predictions = []
        for step in range(samples):
            inputs = motion[:, step] if history_channels == 0 else torch.cat((motion[:, step], history.flatten(1)), dim=1)
            hidden = torch.tanh(inputs @ parameters[0] + parameters[1])
            stiffness = 1.0 + 799.0 * torch.sigmoid(torch.clamp(hidden @ parameters[2] + parameters[3], -50, 50))
            spring = torch.sum(basis[:, step] * stiffness, dim=1)
            predictions.append(spring)
            if history_mode == "target":
                realized = (target[:, step] / 23.5)[:, None]
            elif history_mode == "all":
                realized = torch.stack((target[:, step], spring, target[:, step] - spring), dim=1) / 23.5
            if history_channels:
                history = torch.cat((history[:, 1:], realized.detach()[:, None]), dim=1)
        return torch.stack(predictions, dim=1), stiffness

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction, stiffness = rollout(train_motion, train_basis, train_target)
        loss = torch.mean((prediction - train_target) ** 2)
        loss = loss + 2e-4 * torch.mean(((stiffness - initial) / torch.clamp(initial, min=1)) ** 2)
        loss.backward(); optimizer.step()
    with torch.inference_mode():
        train_prediction, _ = rollout(train_motion, train_basis, train_target)
        test_prediction, _ = rollout(test_motion, test_basis, test_target)
    train_rmse = float(torch.sqrt(torch.mean((train_prediction - train_target) ** 2)).cpu())
    test_rmse = float(torch.sqrt(torch.mean((test_prediction - test_target) ** 2)).cpu())
    baseline = float(torch.sqrt(torch.mean(test_target**2)).cpu())
    result = (train_rmse, test_rmse, baseline, 100 * (baseline - test_rmse) / baseline, input_dim)
    if return_model:
        model = {name: value.detach().cpu().numpy().astype(float) for name, value in zip(("w1", "b1", "w2", "b2"), parameters)}
        return result, model
    return result


def main():
    with np.load(ROLLOUT) as data:
        names = np.asarray(data["joint_names"], dtype=str)
        joint = int(np.flatnonzero(names == JOINT)[0])
        eligible = np.flatnonzero(np.asarray(data["terrain_family"], dtype=str) == FAMILY)
        t = np.asarray(data["time"], dtype=float)
        theta_abs = np.asarray(data["theta"][:, :, joint], dtype=float)
        velocity = np.asarray(data["theta_dot"][:, :, joint], dtype=float)
        target = np.asarray(data["tau_total"][:, :, joint], dtype=float)
    ids = np.random.default_rng(42).permutation(eligible)
    test_ids, train_ids = ids[:5], ids[5:]
    neutral = np.median(theta_abs[:, train_ids]); theta = theta_abs - neutral
    acceleration = np.diff(velocity[:, train_ids], axis=0) / np.diff(t[:, train_ids], axis=0)
    scales = {"theta": np.percentile(np.abs(theta[:, train_ids]), 95),
              "velocity": np.percentile(np.abs(velocity[:, train_ids]), 95),
              "acceleration": np.percentile(np.abs(acceleration), 95)}
    network, _ = load_network(PROJECT_ROOT / "topologies" / "internal_fan_20_spring_model.json")
    angles = np.radians(ANGLE_DEGREES); basis_angles = spring_torque_basis(network, angles, relax_internal=True)
    def basis(ids): return np.stack([interpolate_basis(basis_angles, angles, theta[:, i]) for i in ids])
    train_basis, test_basis = basis(train_ids), basis(test_ids)
    train_target, test_target = target[:, train_ids].T, target[:, test_ids].T
    rows = []
    datasets = {}
    for name, window, acceleration_flag, history in CONFIGS:
        print(f"Running {name}...", flush=True)
        train_motion = motion_features(theta, velocity, t, train_ids, window, acceleration_flag, scales)
        test_motion = motion_features(theta, velocity, t, test_ids, window, acceleration_flag, scales)
        datasets[name] = (train_motion, test_motion, window, history)
        result = run_model(train_motion, test_motion, train_basis, test_basis, train_target, test_target,
                           history, window, initial_stiffnesses(network))
        rows.append((name, window, acceleration_flag, history, *result))
        print(f"  test RMSE {result[1]:.3f} N*m | offload {result[3]:.2f}%", flush=True)
    output = PROJECT_ROOT / "tables" / "slope_motion_window_ablation.csv"
    with output.open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(["configuration", "window", "acceleration", "torque_history",
            "train_rmse_nm", "test_rmse_nm", "baseline_rmse_nm", "rms_offload_pct", "input_dim"]); writer.writerows(rows)
    best = min(rows, key=lambda row: row[5])
    best_name = best[0]
    train_motion, test_motion, best_window, best_history = datasets[best_name]
    print(f"Long training: {best_name}, 20000 iterations...", flush=True)
    long_result, long_model = run_model(
        train_motion, test_motion, train_basis, test_basis, train_target, test_target,
        best_history, best_window, initial_stiffnesses(network), iterations=20000, return_model=True,
    )
    model_path = PROJECT_ROOT / "models" / "internal_fan_go2_fl_knee_slope_motion_only_20000iter.npz"
    save_model(
        model_path, long_model, "internal_fan_go2_fl_knee_slope_motion_only", 1.0, 800.0,
        feature_type="isaac_motion_window_only", source_rollout=str(ROLLOUT), source_joint=JOINT,
        terrain_family=FAMILY, window_size=best_window, input_dim=long_result[4], hidden_dim=128,
        iterations=20000, train_environments=len(train_ids), test_environments=len(test_ids),
        neutral_angle_rad=neutral, theta_scale=scales["theta"], theta_dot_scale=scales["velocity"],
        theta_ddot_scale=scales["acceleration"], train_rmse_nm=long_result[0], test_rmse_nm=long_result[1],
        baseline_test_rmse_nm=long_result[2], held_out_rms_torque_offload_pct=long_result[3],
    )
    print(f"Long result: test RMSE {long_result[1]:.3f} N*m | offload {long_result[3]:.2f}%")
    print(f"Saved model: {model_path}")
    rows.sort(key=lambda row: row[5])
    figure = PROJECT_ROOT / "plots" / "isaaclab_rollouts" / "slope_motion_window_ablation.png"
    fig, axis = plt.subplots(figsize=(10, 5)); axis.bar([r[0] for r in rows], [r[7] for r in rows])
    axis.axhline(0, color="black", linewidth=0.8); axis.set_ylabel("held-out RMS torque offload [%]")
    axis.tick_params(axis="x", rotation=30); axis.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(figure, dpi=180)
    print(f"Saved table: {output}\nSaved plot: {figure}")

if __name__ == "__main__": main()
