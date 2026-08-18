"""Profile-conditioned passive stiffness oracle on genuine 3D mechanics."""

from pathlib import Path
import argparse
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES
from energy_accounting import DEFAULT_MOTORING_EFFICIENCY, DEFAULT_REGEN_EFFICIENCY
from mechanics_3d import load_spatial_topology, torque_and_residual, torque_components
from profile_generator import generate_classified_profile_parameters
from train_profile_conditioned_passive import (
    build_profile_dataset,
    expand_profile_stiffness,
    optimize_profile_stiffness_oracle,
    summary_rows,
    write_rows,
    write_stiffness_rows,
)

DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"
)


def spatial_initial_basis(topology, angles, relaxation_steps):
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(angles), 1)
    components = torque_components(
        topology,
        torch.as_tensor(angles, dtype=torch.float32, device=stiffness.device),
        stiffness,
        relaxation_steps,
    )
    return (
        components / torch.clamp(stiffness, min=1e-9)
    ).detach().cpu().numpy()


def relaxed_spatial_profile_torque(
    dataset,
    topology,
    profile_stiffness,
    relaxation_steps,
    batch_size=1024,
    progress_batches=0,
    progress_label="Exact mechanics",
):
    """Evaluate one constant stiffness vector per profile in full 3D mechanics."""
    schedule = expand_profile_stiffness(
        profile_stiffness, dataset["samples_per_profile"]
    ).reshape(-1, profile_stiffness.shape[1])
    theta = dataset["theta"].reshape(-1)
    torque = np.empty(len(theta), dtype=float)
    force_residual = np.empty(len(theta), dtype=float)
    device = topology["local_positions"].device
    total_batches = int(np.ceil(len(theta) / batch_size))
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, len(theta), batch_size), start=1):
        stop = min(start + batch_size, len(theta))
        theta_batch = torch.as_tensor(
            theta[start:stop], dtype=torch.float32, device=device
        )
        stiffness_batch = torch.as_tensor(
            schedule[start:stop].copy(), dtype=torch.float32, device=device
        )
        values, residual, _ = torque_and_residual(
            topology, theta_batch, stiffness_batch, relaxation_steps
        )
        torque[start:stop] = values.detach().cpu().numpy()
        force_residual[start:stop] = residual.detach().cpu().numpy()
        if progress_batches > 0 and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_batches == 0
        ):
            elapsed = time.perf_counter() - started
            rate = batch_index / max(elapsed, 1e-9)
            eta = (total_batches - batch_index) / max(rate, 1e-9)
            print(
                f"{progress_label}: {batch_index}/{total_batches} batches "
                f"({100.0 * batch_index / total_batches:5.1f}%) | "
                f"elapsed {elapsed / 60.0:6.1f} min | ETA {eta / 60.0:6.1f} min",
                flush=True,
            )
    shape = dataset["target"].shape
    return torque.reshape(shape), force_residual.reshape(shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--profiles-per-family", type=int, default=400)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=30.0)
    parser.add_argument("--min-stiffness", type=float, default=0.0)
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--mechanics-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()
    args.motoring_efficiency = 1.0
    args.regen_efficiency = 0.0
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    topology = load_spatial_topology(args.topology, device)
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, args.relaxation_steps)
    profiles = generate_classified_profile_parameters(
        np.random.default_rng(args.seed), args.profiles_per_family
    )
    dataset = build_profile_dataset(
        profiles, angles, basis, args.duration, args.samples, args.seed + 30_000
    )
    base_k = topology["initial_stiffness"].detach().cpu().numpy()
    stiffness = optimize_profile_stiffness_oracle(
        dataset,
        base_k,
        args.iterations,
        args.learning_rate,
        args.min_stiffness,
        1.0,
        args.energy_weight,
        args.motoring_efficiency,
        args.regen_efficiency,
        device=args.device,
        progress_interval=args.progress_interval,
        unbounded_stiffness=True,
    )
    torque, force_residual = relaxed_spatial_profile_torque(
        dataset,
        topology,
        stiffness,
        args.relaxation_steps,
        args.mechanics_batch_size,
    )
    rows = summary_rows(
        profiles, dataset, torque, stiffness,
        args.motoring_efficiency, args.regen_efficiency
    )
    baseline = sum(row["baseline_motor_work_j"] for row in rows)
    assisted = sum(row["assisted_motor_work_j"] for row in rows)
    offload = 100.0 * (baseline - assisted) / baseline
    name = args.output_name or f"{args.topology.stem}_profile_passive_3d"
    print(f"3D topology: {args.topology.name}")
    print(f"Springs: {len(base_k)}")
    print(f"Exact 3D aggregate offload: {offload:.3f}%")
    print(f"Mean profile offload: {np.mean([r['offload_pct'] for r in rows]):.3f}%")
    print(f"Negative-offload profiles: {100*np.mean([r['offload_pct'] < 0 for r in rows]):.2f}%")
    print(f"Mean final internal-force residual: {np.mean(force_residual):.6f} N")
    print(
        f"Stiffness N/m: median {np.median(stiffness):.3f} | "
        f"p95 {np.percentile(stiffness,95):.3f} | max {np.max(stiffness):.3f}"
    )
    table_dir = PROJECT_ROOT / "tables" / "profile_conditioned_passive_3d"
    write_rows(table_dir / f"{name}_results.csv", rows)
    write_stiffness_rows(table_dir / f"{name}_stiffness.csv", profiles, stiffness)


if __name__ == "__main__":
    main()
