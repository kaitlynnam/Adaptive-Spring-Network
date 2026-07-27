"""Train the internal-fan controller across many Isaac Lab rollout environments."""

from __future__ import annotations

import argparse
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
from train_adaptive_dataset import interpolate_basis, motion_window_features, train_model


def build_dataset(t, theta, theta_dot, target, env_ids, basis_by_angle, angles, window, scales):
    features, basis, targets, times, positions, velocities = [], [], [], [], [], []
    for env_id in env_ids:
        acceleration = np.zeros_like(theta[:, env_id])
        acceleration[1:] = np.diff(theta_dot[:, env_id]) / np.diff(t[:, env_id])
        features.append(motion_window_features(theta[:, env_id], theta_dot[:, env_id], acceleration, window, scales))
        basis.append(interpolate_basis(basis_by_angle, angles, theta[:, env_id]))
        targets.append(target[:, env_id])
        times.append(t[:, env_id])
        positions.append(theta[:, env_id])
        velocities.append(theta_dot[:, env_id])
    samples = t.shape[0]
    return {
        "features": np.vstack(features),
        "basis": np.vstack(basis),
        "target": np.concatenate(targets),
        "t": np.concatenate(times),
        "theta": np.concatenate(positions),
        "theta_dot": np.concatenate(velocities),
        "samples_per_profile": samples,
        "window_size": window,
        "torque_scale": scales["torque"],
    }


def predict_causally(model, dataset, min_k, max_k, device="cpu"):
    samples = dataset["samples_per_profile"]
    profiles = len(dataset["target"]) // samples
    window = dataset["window_size"]
    torch_device = torch.device(device)
    features = torch.as_tensor(dataset["features"].reshape(profiles, samples, -1), dtype=torch.float32, device=torch_device)
    basis = torch.as_tensor(dataset["basis"].reshape(profiles, samples, -1), dtype=torch.float32, device=torch_device)
    target = torch.as_tensor(dataset["target"].reshape(profiles, samples), dtype=torch.float32, device=torch_device)
    parameters = {key: torch.as_tensor(value, dtype=torch.float32, device=torch_device) for key, value in model.items()}
    history = torch.zeros((profiles, window, 3), dtype=torch.float32, device=torch_device)
    predictions = []
    with torch.inference_mode():
        for index in range(samples):
            inputs = torch.cat((features[:, index], history.reshape(profiles, -1)), dim=1)
            hidden = torch.tanh(inputs @ parameters["w1"] + parameters["b1"])
            sigmoid = torch.sigmoid(torch.clamp(hidden @ parameters["w2"] + parameters["b2"], -50, 50))
            stiffness = min_k + (max_k - min_k) * sigmoid
            prediction = torch.sum(basis[:, index] * stiffness, dim=1)
            predictions.append(prediction)
            realized = torch.stack((target[:, index], prediction, target[:, index] - prediction), dim=1)
            history = torch.cat((history[:, 1:], (realized / dataset["torque_scale"])[:, None, :]), dim=1)
    return torch.stack(predictions, dim=1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--joint", default="FL_calf_joint")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--terrain-family", default=None, help="Use only environments with this exported Isaac terrain family.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-name", default="internal_fan_go2_fl_knee_500env")
    parser.add_argument("--load-model", type=Path, default=None, help="Skip optimization and evaluate this saved model.")
    args = parser.parse_args()

    with np.load(args.rollout) as data:
        names = np.asarray(data["joint_names"], dtype=str)
        joint_index = int(np.flatnonzero(names == args.joint)[0])
        t = np.asarray(data["time"], dtype=float)
        theta_absolute = np.asarray(data["theta"][:, :, joint_index], dtype=float)
        theta_dot = np.asarray(data["theta_dot"][:, :, joint_index], dtype=float)
        target = np.asarray(data["tau_total"][:, :, joint_index], dtype=float)
        terrain_families = np.asarray(data["terrain_family"], dtype=str) if "terrain_family" in data.files else None

    env_count = t.shape[1]
    rng = np.random.default_rng(42)
    eligible_ids = np.arange(env_count)
    if args.terrain_family is not None:
        if terrain_families is None:
            raise ValueError("This rollout has no terrain_family array. Re-export it with the updated exporter.")
        available = sorted(set(terrain_families.tolist()))
        eligible_ids = np.flatnonzero(terrain_families == args.terrain_family)
        if len(eligible_ids) < 2:
            raise ValueError(
                f"Terrain family {args.terrain_family!r} has {len(eligible_ids)} environments; "
                f"available families: {', '.join(available)}"
            )
        print(f"Terrain family filter: {args.terrain_family} ({len(eligible_ids)} environments)", flush=True)
    shuffled = rng.permutation(eligible_ids)
    test_count = max(1, round(len(eligible_ids) * args.test_fraction))
    test_ids, train_ids = shuffled[:test_count], shuffled[test_count:]
    neutral = float(np.median(theta_absolute[:, train_ids]))
    theta = theta_absolute - neutral

    dt = np.diff(t[:, train_ids], axis=0)
    acceleration = np.diff(theta_dot[:, train_ids], axis=0) / dt
    def scale(values, fallback):
        return max(float(np.percentile(np.abs(values), 95)), fallback)
    scales = {
        "theta": scale(theta[:, train_ids], np.deg2rad(1)),
        "theta_dot": scale(theta_dot[:, train_ids], 0.1),
        "theta_ddot": scale(acceleration, 0.5),
        "torque": scale(target[:, train_ids], 1.0),
    }

    network, topology = load_network(PROJECT_ROOT / "topologies" / "internal_fan_20_spring_model.json")
    angles = np.radians(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles, relax_internal=True)
    print(f"Building training set: {len(train_ids)} envs x {t.shape[0]} steps", flush=True)
    train = build_dataset(t, theta, theta_dot, target, train_ids, basis_by_angle, angles, args.window_size, scales)
    print(f"Building held-out set: {len(test_ids)} envs x {t.shape[0]} steps", flush=True)
    test = build_dataset(t, theta, theta_dot, target, test_ids, basis_by_angle, angles, args.window_size, scales)

    model_path = PROJECT_ROOT / "models" / f"{args.output_name}.npz"
    if args.load_model:
        with np.load(args.load_model) as saved:
            model = {key: np.asarray(saved[key], dtype=float) for key in ("w1", "b1", "w2", "b2")}
            rmse_key = "train_rmse_nm" if "train_rmse_nm" in saved.files else "optimizer_final_train_rmse_nm"
            train_rmse = float(saved[rmse_key])
        model_path = args.load_model
        print(f"Loaded trained model: {model_path}", flush=True)
    else:
        model, history = train_model(
            train, initial_stiffnesses(network), args.hidden_dim, args.iterations, 0.01,
            1.0, 800.0, 2e-4, 42, 10, args.device, 0.35,
        )
        train_rmse = float(history["train_rmse"][-1])
        print(f"Saving trained model before held-out evaluation: {model_path}", flush=True)
        save_model(
            model_path, model, args.output_name, 1.0, 800.0,
            feature_type="isaac_multi_rollout_motion_torque_window", topology=topology["name"],
            source_rollout=str(args.rollout.resolve()), source_joint=args.joint,
            terrain_family=args.terrain_family or "all",
            train_env_ids=train_ids, test_env_ids=test_ids, train_environments=len(train_ids),
            test_environments=len(test_ids), samples_per_environment=t.shape[0], neutral_angle_rad=neutral,
            window_size=args.window_size, hidden_dim=args.hidden_dim, iterations=args.iterations,
            theta_scale=scales["theta"], theta_dot_scale=scales["theta_dot"],
            theta_ddot_scale=scales["theta_ddot"], torque_scale=scales["torque"],
            optimizer_final_train_rmse_nm=train_rmse,
        )

    print("Calculating causal held-out predictions...", flush=True)
    prediction_device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    test_prediction = predict_causally(model, test, 1.0, 800.0, prediction_device)
    test_target = test["target"].reshape(len(test_ids), -1)
    test_rmse = float(np.sqrt(np.mean((test_prediction - test_target) ** 2)))
    baseline_test_rmse = float(np.sqrt(np.mean(test_target**2)))
    torque_offload = 100 * (baseline_test_rmse - test_rmse) / baseline_test_rmse

    # Replace the preliminary save with complete held-out metrics.
    save_model(
        model_path, model, args.output_name, 1.0, 800.0,
        feature_type="isaac_multi_rollout_motion_torque_window", topology=topology["name"],
        source_rollout=str(args.rollout.resolve()), source_joint=args.joint,
        terrain_family=args.terrain_family or "all",
        train_env_ids=train_ids, test_env_ids=test_ids, train_environments=len(train_ids),
        test_environments=len(test_ids), samples_per_environment=t.shape[0], neutral_angle_rad=neutral,
        window_size=args.window_size, hidden_dim=args.hidden_dim, iterations=args.iterations,
        theta_scale=scales["theta"], theta_dot_scale=scales["theta_dot"],
        theta_ddot_scale=scales["theta_ddot"], torque_scale=scales["torque"],
        train_rmse_nm=train_rmse, test_rmse_nm=test_rmse, baseline_test_rmse_nm=baseline_test_rmse,
        held_out_rms_torque_offload_pct=torque_offload,
    )

    plot_path = PROJECT_ROOT / "plots" / "isaaclab_rollouts" / f"{args.output_name}_held_out_fit.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(t[:, test_ids[0]], test_target[0], "k--", label="Isaac target")
    axis.plot(t[:, test_ids[0]], test_prediction[0], label="internal-fan torque")
    axis.set(xlabel="time [s]", ylabel="torque [N·m]", title=f"Held-out environment {test_ids[0]} — {args.joint}")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(plot_path, dpi=180)

    print(f"Train RMSE: {train_rmse:.6f} N*m")
    print(f"Held-out test RMSE: {test_rmse:.6f} N*m")
    print(f"Held-out zero-assistance RMSE: {baseline_test_rmse:.6f} N*m")
    print(f"Held-out RMS torque offload: {torque_offload:.3f}%")
    print(f"Saved model: {model_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
