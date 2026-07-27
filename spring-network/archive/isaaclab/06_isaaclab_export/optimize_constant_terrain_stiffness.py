"""Optimize one constant internal-fan stiffness vector for a terrain family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "01_core_model"), str(PROJECT_ROOT / "04_adaptive_learning")]

from adaptive_model import ANGLE_DEGREES, initial_stiffnesses, spring_torque_basis
from topology_loader import load_network
from train_adaptive_dataset import interpolate_basis


def burden_power(torque, velocity, motoring_efficiency=0.85, regen_efficiency=0.60):
    power = torque * velocity
    return torch.relu(power) / motoring_efficiency + torch.relu(-power) * (1.0 - regen_efficiency)


def metrics(basis, target, velocity, stiffness):
    spring = np.einsum("esi,i->es", basis, stiffness)
    residual = target - spring
    baseline_rmse = float(np.sqrt(np.mean(target**2)))
    residual_rmse = float(np.sqrt(np.mean(residual**2)))
    torque_offload = 100.0 * (baseline_rmse - residual_rmse) / baseline_rmse

    def burden(torque):
        power = torque * velocity
        return np.maximum(power, 0.0) / 0.85 + np.maximum(-power, 0.0) * 0.40

    baseline_energy = float(np.mean(np.sum(burden(target), axis=1)))
    assisted_energy = float(np.mean(np.sum(burden(residual), axis=1)))
    energy_offload = 100.0 * (baseline_energy - assisted_energy) / baseline_energy
    return {
        "baseline_torque_rmse_nm": baseline_rmse,
        "residual_torque_rmse_nm": residual_rmse,
        "rms_torque_offload_pct": torque_offload,
        "baseline_energy_burden_step_sum": baseline_energy,
        "assisted_energy_burden_step_sum": assisted_energy,
        "energy_offload_pct": energy_offload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--joint", default="FL_calf_joint")
    parser.add_argument("--terrain-family", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--min-stiffness", type=float, default=1.0)
    parser.add_argument("--max-stiffness", type=float, default=800.0)
    parser.add_argument("--stiffness-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    with np.load(args.rollout) as data:
        if "terrain_family" not in data.files:
            raise ValueError("The rollout has no terrain_family labels; use the updated exporter.")
        names = np.asarray(data["joint_names"], dtype=str)
        matches = np.flatnonzero(names == args.joint)
        if not len(matches):
            raise ValueError(f"Unknown joint {args.joint!r}; available: {', '.join(names)}")
        joint = int(matches[0])
        families = np.asarray(data["terrain_family"], dtype=str)
        eligible = np.flatnonzero(families == args.terrain_family)
        available = sorted(set(families.tolist()))
        if len(eligible) < 10:
            raise ValueError(
                f"Need at least 10 environments for {args.terrain_family!r}; found {len(eligible)}. "
                f"Available: {', '.join(available)}"
            )
        theta_absolute = np.asarray(data["theta"][:, :, joint], dtype=float)
        velocity = np.asarray(data["theta_dot"][:, :, joint], dtype=float)
        target = np.asarray(data["tau_total"][:, :, joint], dtype=float)
        dt = float(np.median(np.diff(np.asarray(data["time"][:, eligible[0]], dtype=float))))

    shuffled = np.random.default_rng(args.seed).permutation(eligible)
    test_count = max(1, round(0.1 * len(shuffled)))
    validation_count = max(1, round(0.1 * len(shuffled)))
    test_ids = shuffled[:test_count]
    validation_ids = shuffled[test_count : test_count + validation_count]
    train_ids = shuffled[test_count + validation_count :]
    neutral_angle = float(np.median(theta_absolute[:, train_ids]))
    theta = theta_absolute - neutral_angle

    network, topology = load_network(PROJECT_ROOT / "topologies" / "internal_fan_20_spring_model.json")
    angles = np.radians(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles, relax_internal=True)

    def subset(ids):
        basis = np.stack([interpolate_basis(basis_by_angle, angles, theta[:, env_id]) for env_id in ids])
        return basis.astype(np.float32), target[:, ids].T.astype(np.float32), velocity[:, ids].T.astype(np.float32)

    train_basis, train_target, train_velocity = subset(train_ids)
    validation_basis, validation_target, validation_velocity = subset(validation_ids)
    test_basis, test_target, test_velocity = subset(test_ids)

    device = torch.device(args.device)
    tensors = [
        torch.as_tensor(value, device=device)
        for value in (train_basis, train_target, train_velocity, validation_basis, validation_target, validation_velocity)
    ]
    train_basis_t, train_target_t, train_velocity_t, validation_basis_t, validation_target_t, validation_velocity_t = tensors
    initial_k = initial_stiffnesses(network)
    scaled = np.clip((initial_k - args.min_stiffness) / (args.max_stiffness - args.min_stiffness), 1e-6, 1 - 1e-6)
    logits = torch.nn.Parameter(torch.as_tensor(np.log(scaled / (1 - scaled)), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([logits], lr=args.learning_rate)
    initial_tensor = torch.as_tensor(initial_k, dtype=torch.float32, device=device)
    baseline_train_burden = torch.mean(burden_power(train_target_t, train_velocity_t)).detach()
    best_validation = float("inf")
    best_stiffness = initial_k.copy()
    best_iteration = 0

    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        stiffness = args.min_stiffness + (args.max_stiffness - args.min_stiffness) * torch.sigmoid(logits)
        spring = torch.einsum("esi,i->es", train_basis_t, stiffness)
        residual = train_target_t - spring
        energy_ratio = torch.mean(burden_power(residual, train_velocity_t)) / torch.clamp(baseline_train_burden, min=1e-9)
        regularization = args.stiffness_weight * torch.mean(((stiffness - initial_tensor) / torch.clamp(initial_tensor, min=1.0)) ** 2)
        loss = energy_ratio + regularization
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            stiffness = args.min_stiffness + (args.max_stiffness - args.min_stiffness) * torch.sigmoid(logits)
            validation_residual = validation_target_t - torch.einsum("esi,i->es", validation_basis_t, stiffness)
            validation_burden = float(torch.mean(burden_power(validation_residual, validation_velocity_t)).cpu())
            if validation_burden < best_validation:
                best_validation = validation_burden
                best_stiffness = stiffness.cpu().numpy().astype(float).copy()
                best_iteration = iteration
        if iteration == 1 or iteration % 500 == 0 or iteration == args.iterations:
            print(
                f"iteration {iteration:5d} | train energy offload {(1-float(energy_ratio.detach().cpu()))*100:7.2f}% "
                f"| best validation iteration {best_iteration}",
                flush=True,
            )

    results = {
        "train": metrics(train_basis, train_target, train_velocity, best_stiffness),
        "validation": metrics(validation_basis, validation_target, validation_velocity, best_stiffness),
        "test": metrics(test_basis, test_target, test_velocity, best_stiffness),
    }
    output_name = args.output_name or f"constant_internal_fan_{args.joint}_{args.terrain_family}"
    model_path = PROJECT_ROOT / "models" / f"{output_name}.npz"
    report_path = PROJECT_ROOT / "tables" / f"{output_name}_metrics.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_path,
        model_type="constant_terrain_stiffness",
        topology=topology["name"],
        source_rollout=str(args.rollout.resolve()),
        joint=args.joint,
        terrain_family=args.terrain_family,
        stiffness=best_stiffness,
        neutral_angle_rad=neutral_angle,
        best_iteration=best_iteration,
        dt=dt,
        train_env_ids=train_ids,
        validation_env_ids=validation_ids,
        test_env_ids=test_ids,
    )
    report = {
        "model": str(model_path),
        "terrain_family": args.terrain_family,
        "joint": args.joint,
        "train_environments": len(train_ids),
        "validation_environments": len(validation_ids),
        "test_environments": len(test_ids),
        "best_iteration": best_iteration,
        "stiffness_min_n_per_m": float(best_stiffness.min()),
        "stiffness_max_n_per_m": float(best_stiffness.max()),
        **results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {report_path}")


if __name__ == "__main__":
    main()
