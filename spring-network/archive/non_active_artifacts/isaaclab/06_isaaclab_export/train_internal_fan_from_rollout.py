"""Train the internal-fan stiffness controller on one Isaac Lab joint rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES, initial_stiffnesses, save_model, spring_torque_basis
from topology_loader import load_network
from train_adaptive_dataset import (
    causal_derivative,
    interpolate_basis,
    motion_window_features,
    train_model,
)


def causal_prediction(model, dataset, min_k, max_k):
    samples = dataset["samples_per_profile"]
    window_size = dataset["window_size"]
    torque_scale = dataset["torque_scale"]
    history = np.zeros((window_size, 3), dtype=float)
    predictions = []
    stiffness_rows = []
    for index in range(samples):
        inputs = np.concatenate((dataset["features"][index], history.reshape(-1)))
        hidden = np.tanh(inputs @ model["w1"] + model["b1"])
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(hidden @ model["w2"] + model["b2"], -50.0, 50.0)))
        stiffness = min_k + (max_k - min_k) * sigmoid
        spring_torque = float(dataset["basis"][index] @ stiffness)
        motor_torque = dataset["target"][index] - spring_torque
        realized = np.array([dataset["target"][index], spring_torque, motor_torque]) / torque_scale
        history = np.vstack((history[1:], realized))
        predictions.append(spring_torque)
        stiffness_rows.append(stiffness)
    return np.asarray(predictions), np.asarray(stiffness_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--joint", default="FL_calf_joint")
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-name", default="internal_fan_go2_fl_knee_100step")
    args = parser.parse_args()

    with np.load(args.rollout) as data:
        names = np.asarray(data["joint_names"], dtype=str)
        matches = np.flatnonzero(names == args.joint)
        if not len(matches):
            raise ValueError(f"Unknown joint {args.joint!r}; available: {', '.join(names)}")
        joint_index = int(matches[0])
        t = np.asarray(data["time"][:, args.env_id], dtype=float)
        theta_absolute = np.asarray(data["theta"][:, args.env_id, joint_index], dtype=float)
        theta_dot = np.asarray(data["theta_dot"][:, args.env_id, joint_index], dtype=float)
        target = np.asarray(data["tau_total"][:, args.env_id, joint_index], dtype=float)

    if len(t) != 100:
        raise ValueError(f"This experiment requires exactly 100 rollout samples; found {len(t)}")

    neutral_angle = float(np.median(theta_absolute))
    theta = theta_absolute - neutral_angle
    theta_ddot = causal_derivative(theta_dot, t)

    def scale(values, fallback):
        return max(float(np.percentile(np.abs(values), 95)), fallback)

    scales = {
        "theta": scale(theta, np.deg2rad(1.0)),
        "theta_dot": scale(theta_dot, 0.1),
        "theta_ddot": scale(theta_ddot, 0.5),
        "torque": scale(target, 1.0),
    }
    topology_path = PROJECT_ROOT / "topologies" / "internal_fan_20_spring_model.json"
    network, topology = load_network(topology_path)
    angles_rad = np.radians(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles_rad, relax_internal=True)
    dataset = {
        "features": motion_window_features(theta, theta_dot, theta_ddot, args.window_size, scales),
        "target": target,
        "basis": interpolate_basis(basis_by_angle, angles_rad, theta),
        "profile_indices": np.zeros(len(t), dtype=int),
        "t": t,
        "theta": theta,
        "theta_dot": theta_dot,
        "theta_ddot": theta_ddot,
        "samples_per_profile": len(t),
        "window_size": args.window_size,
        "torque_scale": scales["torque"],
    }

    model, history = train_model(
        dataset=dataset,
        initial_k=initial_stiffnesses(network),
        hidden_dim=args.hidden_dim,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        min_k=1.0,
        max_k=800.0,
        stiffness_weight=2e-4,
        seed=42,
        progress_interval=100,
        device=args.device,
        energy_weight=0.35,
    )
    prediction, stiffness = causal_prediction(model, dataset, 1.0, 800.0)
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))

    model_path = PROJECT_ROOT / "models" / f"{args.output_name}.npz"
    save_model(
        model_path,
        model,
        args.output_name,
        1.0,
        800.0,
        feature_type="isaac_rollout_motion_torque_window",
        topology=topology["name"],
        source_rollout=str(args.rollout.resolve()),
        source_joint=args.joint,
        source_env=args.env_id,
        samples=len(t),
        neutral_angle_rad=neutral_angle,
        window_size=args.window_size,
        theta_scale=scales["theta"],
        theta_dot_scale=scales["theta_dot"],
        theta_ddot_scale=scales["theta_ddot"],
        torque_scale=scales["torque"],
        hidden_dim=args.hidden_dim,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        training_rmse_nm=rmse,
    )

    plot_path = PROJECT_ROOT / "plots" / "isaaclab_rollouts" / f"{args.output_name}_fit.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, target, "k--", label="Isaac target torque")
    axes[0].plot(t, prediction, label="internal-fan spring torque")
    axes[0].set_ylabel("torque [N·m]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, np.rad2deg(theta))
    axes[1].set_ylabel("relative knee angle [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Internal-fan training fit: {args.joint}, env {args.env_id}\n100 training samples; RMSE {rmse:.3f} N·m")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")

    print(f"Topology: {topology['name']} ({len(network.springs)} springs)")
    print(f"Joint: {args.joint} | samples: {len(t)} | neutral angle: {np.rad2deg(neutral_angle):.3f} deg")
    print(f"Training-fit RMSE: {rmse:.6f} N*m")
    print(f"Stiffness range used: {stiffness.min():.3f} to {stiffness.max():.3f} N/m")
    print(f"Saved model: {model_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
