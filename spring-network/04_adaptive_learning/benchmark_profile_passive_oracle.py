"""Measure the per-profile stiffness ceiling on arbitrary torque-angle curves."""

from pathlib import Path
import argparse
import csv
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES, initial_stiffnesses, spring_torque_basis
from energy_accounting import DEFAULT_MOTORING_EFFICIENCY, DEFAULT_REGEN_EFFICIENCY
from profile_generator import generate_classified_profile_parameters
from topology_loader import load_network
from train_adaptive_dataset import TRAINING_NETWORK_PRESETS
from train_profile_conditioned_passive import (
    build_profile_dataset,
    optimize_profile_stiffness_oracle,
    refresh_profile_torque_basis,
    relaxed_profile_torque,
    summary_rows,
    write_rows,
    write_stiffness_rows,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=sorted(TRAINING_NETWORK_PRESETS), default="fan")
    parser.add_argument("--topology", default=None)
    parser.add_argument("--profiles-per-family", type=int, default=400)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-stiffness", type=float, default=1.0)
    parser.add_argument("--max-stiffness", type=float, default=800.0)
    parser.add_argument(
        "--unbounded-stiffness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use positive softplus stiffness with no upper cap.",
    )
    parser.add_argument("--energy-weight", type=float, default=10.0)
    parser.add_argument("--motoring-efficiency", type=float, default=DEFAULT_MOTORING_EFFICIENCY)
    parser.add_argument("--regen-efficiency", type=float, default=DEFAULT_REGEN_EFFICIENCY)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--mechanics-batch-size", type=int, default=4096)
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--surrogate-refreshes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-name", default="profile_passive_oracle")
    args = parser.parse_args()
    preset = TRAINING_NETWORK_PRESETS[args.network]
    topology_path = Path(args.topology) if args.topology else Path(preset["topology"])
    network, _ = load_network(topology_path)
    angles = np.radians(ANGLE_DEGREES)
    basis = spring_torque_basis(network, angles, relax_internal=True)
    profiles = generate_classified_profile_parameters(
        np.random.default_rng(args.seed), args.profiles_per_family
    )
    dataset = build_profile_dataset(
        profiles, angles, basis, args.duration, args.samples, args.seed + 30_000
    )
    active_dataset = dataset
    stiffness = None
    for phase in range(args.surrogate_refreshes + 1):
        stiffness = optimize_profile_stiffness_oracle(
            active_dataset,
            initial_stiffnesses(network) if stiffness is None else stiffness,
            args.iterations,
            args.learning_rate,
            args.min_stiffness,
            args.max_stiffness,
            args.energy_weight,
            args.motoring_efficiency,
            args.regen_efficiency,
            device=args.device,
            progress_interval=args.progress_interval,
            unbounded_stiffness=args.unbounded_stiffness,
        )
        if phase < args.surrogate_refreshes:
            print(f"Refreshing exact local torque basis ({phase + 1}/{args.surrogate_refreshes})...")
            active_dataset = refresh_profile_torque_basis(
                dataset,
                topology_path,
                stiffness,
                device=args.device,
                batch_size=min(args.mechanics_batch_size, 1024),
                relaxation_steps=args.relaxation_steps,
            )
    torque = relaxed_profile_torque(
        dataset, topology_path, stiffness, args.device,
        args.mechanics_batch_size, args.relaxation_steps
    )
    rows = summary_rows(
        profiles, dataset, torque, stiffness,
        args.motoring_efficiency, args.regen_efficiency
    )
    baseline = sum(row["baseline_energy_burden_j"] for row in rows)
    assisted = sum(row["assisted_energy_burden_j"] for row in rows)
    aggregate_offload = 100.0 * (baseline - assisted) / baseline
    print(f"Exact-mechanics oracle aggregate offload: {aggregate_offload:.3f}%")
    print(f"Exact-mechanics oracle mean profile offload: {np.mean([r['offload_pct'] for r in rows]):.3f}%")
    print(f"Negative-offload profiles: {100.0 * np.mean([r['offload_pct'] < 0 for r in rows]):.2f}%")
    print(
        "Oracle stiffness N/m: "
        f"median {np.median(stiffness):.3f} | "
        f"p95 {np.percentile(stiffness, 95):.3f} | "
        f"p99 {np.percentile(stiffness, 99):.3f} | "
        f"max {np.max(stiffness):.3f}"
    )
    table_dir = PROJECT_ROOT / "tables" / "profile_conditioned_passive"
    write_rows(table_dir / f"{args.output_name}_results.csv", rows)
    write_stiffness_rows(
        table_dir / f"{args.output_name}_stiffness.csv", profiles, stiffness
    )


if __name__ == "__main__":
    main()
