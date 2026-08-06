"""Calculate torque and energy offload for a rollout-trained internal-fan model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "01_core_model"),
    str(PROJECT_ROOT / "04_adaptive_learning"),
    str(PROJECT_ROOT / "06_isaaclab_export"),
]

from adaptive_model import ANGLE_DEGREES, spring_torque_basis
from energy_accounting import numpy_power_accounting
from topology_loader import load_network
from train_adaptive_dataset import causal_derivative, integrate_trapezoid, interpolate_basis, motion_window_features
from train_internal_fan_from_rollout import causal_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    with np.load(args.model) as saved:
        model = {name: np.asarray(saved[name], dtype=float) for name in ("w1", "b1", "w2", "b2")}
        joint = str(saved["source_joint"])
        env_id = int(saved["source_env"])
        neutral = float(saved["neutral_angle_rad"])
        window = int(saved["window_size"])
        scales = {key: float(saved[f"{key}_scale"]) for key in ("theta", "theta_dot", "theta_ddot", "torque")}
        min_k, max_k = float(saved["min_k"]), float(saved["max_k"])

    with np.load(args.rollout) as data:
        names = np.asarray(data["joint_names"], dtype=str)
        joint_index = int(np.flatnonzero(names == joint)[0])
        t = np.asarray(data["time"][:, env_id], dtype=float)
        theta = np.asarray(data["theta"][:, env_id, joint_index], dtype=float) - neutral
        theta_dot = np.asarray(data["theta_dot"][:, env_id, joint_index], dtype=float)
        target = np.asarray(data["tau_total"][:, env_id, joint_index], dtype=float)

    network, _ = load_network(PROJECT_ROOT / "topologies" / "internal_fan_20_spring_model.json")
    angles = np.radians(ANGLE_DEGREES)
    dataset = {
        "features": motion_window_features(theta, theta_dot, causal_derivative(theta_dot, t), window, scales),
        "basis": interpolate_basis(spring_torque_basis(network, angles, relax_internal=True), angles, theta),
        "target": target,
        "samples_per_profile": len(t),
        "window_size": window,
        "torque_scale": scales["torque"],
    }
    spring_torque, _ = causal_prediction(model, dataset, min_k, max_k)
    residual = target - spring_torque

    baseline_rmse = float(np.sqrt(np.mean(target**2)))
    residual_rmse = float(np.sqrt(np.mean(residual**2)))
    rms_offload = 100.0 * (baseline_rmse - residual_rmse) / baseline_rmse

    baseline = numpy_power_accounting(target * theta_dot, 0.85, 0.60)
    assisted = numpy_power_accounting(residual * theta_dot, 0.85, 0.60)
    baseline_energy = float(integrate_trapezoid(baseline["energy_burden_power"], t))
    assisted_energy = float(integrate_trapezoid(assisted["energy_burden_power"], t))
    energy_offload = 100.0 * (baseline_energy - assisted_energy) / baseline_energy

    print(f"joint={joint} env={env_id} samples={len(t)}")
    print(f"baseline_torque_rmse_nm={baseline_rmse:.6f}")
    print(f"residual_torque_rmse_nm={residual_rmse:.6f}")
    print(f"rms_torque_offload_pct={rms_offload:.6f}")
    print(f"baseline_energy_burden_j={baseline_energy:.6f}")
    print(f"assisted_energy_burden_j={assisted_energy:.6f}")
    print(f"energy_offload_pct={energy_offload:.6f}")


if __name__ == "__main__":
    main()
