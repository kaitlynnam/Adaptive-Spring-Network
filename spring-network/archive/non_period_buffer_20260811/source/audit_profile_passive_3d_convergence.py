"""Audit independent and warm-started equilibrium on learned MLP stiffnesses."""

from pathlib import Path
import argparse
import csv
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES
from benchmark_profile_passive_3d import relaxed_spatial_profile_torque, spatial_initial_basis
from mechanics_3d import load_spatial_topology, torque_and_residual
from profile_generator import generate_profile_parameters
from train_profile_conditioned_passive import (
    build_profile_dataset,
    predict_profile_stiffness,
    summary_rows,
)


def warm_started_torque(
    dataset, topology, stiffness, max_steps, force_tolerance
):
    profiles, samples = dataset["theta"].shape
    torque = np.empty((profiles, samples), dtype=float)
    residual = np.empty_like(torque)
    iterations = []
    internal = topology["internal_indices"]
    previous_internal = None
    device = topology["local_positions"].device
    for sample in range(samples):
        theta = torch.as_tensor(
            dataset["theta"][:, sample], dtype=torch.float32, device=device
        )
        stiffness_tensor = torch.as_tensor(
            stiffness, dtype=torch.float32, device=device
        )
        values, force_residual, positions, completed = torque_and_residual(
            topology,
            theta,
            stiffness_tensor,
            max_steps,
            initial_internal=previous_internal,
            force_tolerance=force_tolerance,
            return_iterations=True,
        )
        torque[:, sample] = values.detach().cpu().numpy()
        residual[:, sample] = force_residual.detach().cpu().numpy()
        previous_internal = positions[:, internal, :].detach()
        iterations.append(completed)
    return torque, residual, np.asarray(iterations)


def aggregate(name, profiles, dataset, torque, stiffness, residual, seconds, iterations=None):
    rows = summary_rows(profiles, dataset, torque, stiffness)
    baseline = sum(row["baseline_motor_work_j"] for row in rows)
    assisted = sum(row["assisted_motor_work_j"] for row in rows)
    return {
        "solver": name,
        "aggregate_offload_pct": 100.0 * (baseline - assisted) / baseline,
        "mean_profile_rmse_nm": float(np.mean([row["residual_rmse_nm"] for row in rows])),
        "mean_force_residual_n": float(np.mean(residual)),
        "max_force_residual_n": float(np.max(residual)),
        "mean_solver_steps": float(np.mean(iterations)) if iterations is not None else "",
        "max_solver_steps": int(np.max(iterations)) if iterations is not None else "",
        "runtime_seconds": seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--training-profiles", type=int, default=6000)
    parser.add_argument("--test-profiles", type=int, default=1200)
    parser.add_argument("--audit-profiles", type=int, default=90)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--force-tolerance", type=float, default=1e-3)
    parser.add_argument("--warm-max-steps", type=int, default=1000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    saved = np.load(args.checkpoint, allow_pickle=False)
    model = {name: saved[name] for name in ("w1", "b1", "w2", "b2")}
    seed = int(saved["seed"])
    topology_path = Path(str(saved["topology"]))
    if not topology_path.is_absolute():
        topology_path = Path.cwd() / topology_path
    topology = load_spatial_topology(topology_path, args.device)
    topology["nonlinear_power"] = int(saved["nonlinear_power"]) if "nonlinear_power" in saved else 1
    topology["nonlinear_ratio"] = float(saved["nonlinear_ratio"]) if "nonlinear_ratio" in saved else 0.0
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, 300)
    rng = np.random.default_rng(seed)
    generate_profile_parameters(rng, args.training_profiles)
    all_test = generate_profile_parameters(rng, args.test_profiles)
    selected = all_test[:args.audit_profiles]
    dataset = build_profile_dataset(
        selected, angles, basis, args.duration, args.samples, seed + 30_000
    )
    stiffness = predict_profile_stiffness(
        model, dataset, float(saved["min_k"]), 1.0, unbounded_stiffness=True
    )
    results = []
    torque_by_name = {}
    for steps in (300, 600, 1000):
        start = time.perf_counter()
        torque, residual = relaxed_spatial_profile_torque(
            dataset, topology, stiffness, steps, args.batch_size
        )
        elapsed = time.perf_counter() - start
        name = f"independent_{steps}"
        torque_by_name[name] = torque
        results.append(
            aggregate(name, selected, dataset, torque, stiffness, residual, elapsed)
        )
    start = time.perf_counter()
    torque, residual, iterations = warm_started_torque(
        dataset,
        topology,
        stiffness,
        args.warm_max_steps,
        args.force_tolerance,
    )
    elapsed = time.perf_counter() - start
    torque_by_name["warm_tolerance"] = torque
    results.append(
        aggregate(
            "warm_tolerance",
            selected,
            dataset,
            torque,
            stiffness,
            residual,
            elapsed,
            iterations,
        )
    )
    reference = torque_by_name["independent_1000"]
    for row in results:
        difference = torque_by_name[row["solver"]] - reference
        row["torque_rmse_vs_independent_1000_nm"] = float(
            np.sqrt(np.mean(difference**2))
        )
        row["max_torque_difference_nm"] = float(np.max(np.abs(difference)))
        print(row)
    output = (
        PROJECT_ROOT
        / "tables"
        / "profile_conditioned_passive_3d"
        / f"{args.checkpoint.stem}_convergence_audit.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
