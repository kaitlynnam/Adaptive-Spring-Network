"""Compare fixed and periodicity-controlled internal-fan stiffness schedules.

This is an isolated experiment and does not alter the existing adaptive model.
Stiffness updates occur once per cycle. The controlled schedule blends toward
the best current-cycle stiffness using a class-dependent update fraction.
"""

from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES, spring_torque_basis
from experiment_cycle_periodicity import interpolate_basis, make_sequence
from periodicity_classifier import periodicity_score
from profile_generator import generate_profile_parameters
from topology_loader import load_network


def bounded_fit(basis, target, min_k, max_k):
    return lsq_linear(
        basis, target, bounds=(min_k, max_k), lsmr_tol="auto", max_iter=300
    ).x


def evaluate_schedule(t, theta, target, basis, stiffness):
    predicted = np.sum(basis * stiffness, axis=1)
    residual = target - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-9)
    torque_offload = 100.0 * (1.0 - rmse / target_rms)
    theta_dot = np.gradient(theta, t)
    before = float(np.mean(np.maximum(target * theta_dot, 0.0)))
    after = float(np.mean(np.maximum(residual * theta_dot, 0.0)))
    power_offload = 100.0 * (before - after) / max(before, 1e-9)
    changes = np.diff(stiffness, axis=0)
    mean_step_change = float(np.mean(np.abs(changes))) if len(changes) else 0.0
    total_change = float(np.sum(np.abs(changes)))
    return rmse, torque_offload, power_offload, mean_step_change, total_change


def make_schedules(basis, target, cycles, samples_per_cycle, min_k, max_k, alpha):
    fixed_k = bounded_fit(basis, target, min_k, max_k)
    fixed = np.repeat(fixed_k[None, :], len(target), axis=0)

    controlled = np.empty_like(basis)
    unrestricted = np.empty_like(basis)
    previous = fixed_k.copy()
    for cycle in range(cycles):
        start = cycle * samples_per_cycle
        stop = start + samples_per_cycle
        cycle_k = bounded_fit(basis[start:stop], target[start:stop], min_k, max_k)
        unrestricted[start:stop] = cycle_k
        previous = np.clip(previous + alpha * (cycle_k - previous), min_k, max_k)
        controlled[start:stop] = previous
    return {"fixed": fixed, "controlled": controlled, "per_cycle_upper_bound": unrestricted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--samples-per-cycle", type=int, default=80)
    parser.add_argument("--high-variation-nm", type=float, default=2.0)
    parser.add_argument("--low-variation-nm", type=float, default=35.0)
    parser.add_argument("--high-update-fraction", type=float, default=0.10)
    parser.add_argument("--low-update-fraction", type=float, default=0.80)
    parser.add_argument("--min-stiffness", type=float, default=1.0)
    parser.add_argument("--max-stiffness", type=float, default=800.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output-name", default="periodicity_controlled_adaptation_test")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    profiles = generate_profile_parameters(rng, args.cases)
    network, _ = load_network(PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "internal_fan_20_spring_model.json")
    angles = np.deg2rad(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles, relax_internal=True)
    rows = []

    conditions = (
        ("high_periodicity", args.high_variation_nm, args.high_update_fraction),
        ("low_periodicity", args.low_variation_nm, args.low_update_fraction),
    )
    for case_index, profile in enumerate(profiles):
        perturbations = rng.normal(size=(args.cycles, 5))
        perturbations -= np.mean(perturbations, axis=0, keepdims=True)
        for label, magnitude, alpha in conditions:
            t, theta, target = make_sequence(
                profile, perturbations, magnitude, args.cycles, args.samples_per_cycle
            )
            basis = interpolate_basis(basis_by_angle, angles, theta)
            score = periodicity_score(t, theta, target, nominal_frequency_hz=1.0)
            schedules = make_schedules(
                basis,
                target,
                args.cycles,
                args.samples_per_cycle,
                args.min_stiffness,
                args.max_stiffness,
                alpha,
            )
            for mode, stiffness in schedules.items():
                rmse, torque_offload, power_offload, mean_change, total_change = evaluate_schedule(
                    t, theta, target, basis, stiffness
                )
                rows.append(
                    {
                        "case": case_index,
                        "class": label,
                        "mode": mode,
                        "variation_nm": magnitude,
                        "update_fraction": 0.0 if mode == "fixed" else (1.0 if mode == "per_cycle_upper_bound" else alpha),
                        "periodicity_score": score["periodicity_score"],
                        "rmse_nm": rmse,
                        "torque_offload_pct": torque_offload,
                        "positive_power_offload_pct": power_offload,
                        "mean_abs_stiffness_step": mean_change,
                        "total_abs_stiffness_change": total_change,
                    }
                )

    table_path = PROJECT_ROOT / "tables" / "legacy" / f"{args.output_name}.csv"
    plot_path = PROJECT_ROOT / "plots" / "legacy" / "dataset_examples" / f"{args.output_name}.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    modes = ("fixed", "controlled", "per_cycle_upper_bound")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, label in enumerate(("high_periodicity", "low_periodicity")):
        selected = [row for row in rows if row["class"] == label]
        axes[row_index, 0].boxplot(
            [[r["rmse_nm"] for r in selected if r["mode"] == mode] for mode in modes],
            tick_labels=["fixed", "controlled", "upper"],
        )
        axes[row_index, 0].set_ylabel("RMSE [Nm]")
        axes[row_index, 1].boxplot(
            [[r["positive_power_offload_pct"] for r in selected if r["mode"] == mode] for mode in modes],
            tick_labels=["fixed", "controlled", "upper"],
        )
        axes[row_index, 1].set_ylabel("Positive-power offload [%]")
        for ax in axes[row_index]:
            ax.set_title(label)
            ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("Periodicity-controlled stiffness experiment")
    for label, _, alpha in conditions:
        print(f"\n{label} (controlled update fraction {alpha:.2f})")
        for mode in modes:
            selected = [r for r in rows if r["class"] == label and r["mode"] == mode]
            print(
                f"  {mode:21s} | RMSE {np.mean([r['rmse_nm'] for r in selected]):7.3f} Nm | "
                f"power offload {np.mean([r['positive_power_offload_pct'] for r in selected]):7.2f}% | "
                f"total |dk| {np.mean([r['total_abs_stiffness_change'] for r in selected]):10.1f}"
            )
    print(f"\nSaved table: {table_path}")
    print(f"Saved plot:  {plot_path}")


if __name__ == "__main__":
    main()
