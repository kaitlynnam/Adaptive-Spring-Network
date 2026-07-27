"""Standalone test of whether repeated torque-angle cycles are easier to offload.

This experiment does not modify or replace the existing adaptive training path.
For each matched case it creates one base five-knot torque-angle graph, then
perturbs its torque knots between cycles. The high- and low-periodicity cases
use exactly the same perturbation directions but different magnitudes. One
constant internal-fan stiffness vector is fitted across the entire sequence.
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
from periodicity_classifier import periodicity_score
from profile_generator import ANGLE_LIMIT_RAD, generate_profile_parameters
from topology_loader import load_network


def interpolate_basis(basis_by_angle, angles, theta):
    return np.column_stack(
        [np.interp(theta, angles, basis_by_angle[:, index]) for index in range(basis_by_angle.shape[1])]
    )


def make_sequence(base_profile, perturbations, perturbation_nm, cycles, samples_per_cycle):
    phase = np.arange(samples_per_cycle, dtype=float) / samples_per_cycle
    theta_cycle = ANGLE_LIMIT_RAD * np.sin(2.0 * np.pi * phase)
    theta = np.tile(theta_cycle, cycles)
    t = np.arange(len(theta), dtype=float) / samples_per_cycle
    torque = np.empty_like(theta)
    for cycle in range(cycles):
        start = cycle * samples_per_cycle
        stop = start + samples_per_cycle
        knots_tau = base_profile["knots_tau"] + perturbation_nm * perturbations[cycle]
        torque[start:stop] = np.interp(
            theta_cycle,
            base_profile["knots_theta"],
            knots_tau,
            left=knots_tau[0],
            right=knots_tau[-1],
        )
    return t, theta, torque


def fit_case(basis_by_angle, angles, t, theta, target, min_k, max_k):
    basis = interpolate_basis(basis_by_angle, angles, theta)
    result = lsq_linear(basis, target, bounds=(min_k, max_k), lsmr_tol="auto", max_iter=300)
    predicted = basis @ result.x
    rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
    target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-9)
    torque_offload = 100.0 * (1.0 - rmse / target_rms)

    theta_dot = np.gradient(theta, t)
    before = float(np.mean(np.maximum(target * theta_dot, 0.0)))
    after = float(np.mean(np.maximum((target - predicted) * theta_dot, 0.0)))
    power_offload = 100.0 * (before - after) / max(before, 1e-9)
    return rmse, torque_offload, power_offload, result.x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--samples-per-cycle", type=int, default=80)
    parser.add_argument("--high-variation-nm", type=float, default=2.0)
    parser.add_argument("--low-variation-nm", type=float, default=35.0)
    parser.add_argument("--min-stiffness", type=float, default=1.0)
    parser.add_argument("--max-stiffness", type=float, default=800.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output-name", default="cycle_periodicity_fixed_fan_test")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    profiles = generate_profile_parameters(rng, args.cases)
    network, _ = load_network(PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "internal_fan_20_spring_model.json")
    angles = np.deg2rad(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles, relax_internal=True)

    rows = []
    for case_index, profile in enumerate(profiles):
        perturbations = rng.normal(size=(args.cycles, 5))
        perturbations -= np.mean(perturbations, axis=0, keepdims=True)
        for label, magnitude in (
            ("high_periodicity", args.high_variation_nm),
            ("low_periodicity", args.low_variation_nm),
        ):
            t, theta, target = make_sequence(
                profile, perturbations, magnitude, args.cycles, args.samples_per_cycle
            )
            score = periodicity_score(t, theta, target, nominal_frequency_hz=1.0)
            rmse, torque_offload, power_offload, stiffness = fit_case(
                basis_by_angle,
                angles,
                t,
                theta,
                target,
                args.min_stiffness,
                args.max_stiffness,
            )
            rows.append(
                {
                    "case": case_index,
                    "class": label,
                    "variation_nm": magnitude,
                    "periodicity_score": score["periodicity_score"],
                    "cycle_error": score["cycle_repeatability_error"],
                    "rmse_nm": rmse,
                    "torque_offload_pct": torque_offload,
                    "positive_power_offload_pct": power_offload,
                    "mean_stiffness": float(np.mean(stiffness)),
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

    labels = ("high_periodicity", "low_periodicity")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fields = ("periodicity_score", "rmse_nm", "positive_power_offload_pct")
    titles = ("Periodicity score", "Fixed-fan RMSE [Nm]", "Positive-power offload [%]")
    for ax, field, title in zip(axes, fields, titles):
        ax.boxplot([[row[field] for row in rows if row["class"] == label] for label in labels], tick_labels=["high", "low"])
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("Matched fixed-stiffness internal-fan experiment")
    for label in labels:
        selected = [row for row in rows if row["class"] == label]
        print(
            f"{label:17s} | score {np.mean([r['periodicity_score'] for r in selected]):.4f} | "
            f"RMSE {np.mean([r['rmse_nm'] for r in selected]):.3f} Nm | "
            f"positive-power offload {np.mean([r['positive_power_offload_pct'] for r in selected]):.2f}%"
        )
    print(f"Saved table: {table_path}")
    print(f"Saved plot:  {plot_path}")


if __name__ == "__main__":
    main()
