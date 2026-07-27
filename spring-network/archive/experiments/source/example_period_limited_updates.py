"""Visual example of internal-fan updates limited to nominal gait periods."""

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES, spring_torque_basis
from experiment_cycle_periodicity import interpolate_basis, make_sequence
from experiment_periodicity_adaptation import evaluate_schedule, make_schedules
from periodicity_classifier import periodicity_score
from profile_generator import generate_profile_parameters
from topology_loader import load_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--samples-per-cycle", type=int, default=80)
    parser.add_argument("--variation-nm", type=float, default=35.0)
    parser.add_argument("--update-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--output-name", default="period_limited_update_example")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    profile = generate_profile_parameters(rng, 1)[0]
    perturbations = rng.normal(size=(args.cycles, 5))
    perturbations -= np.mean(perturbations, axis=0, keepdims=True)
    t, theta, target = make_sequence(
        profile,
        perturbations,
        args.variation_nm,
        args.cycles,
        args.samples_per_cycle,
    )

    network, _ = load_network(PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "internal_fan_20_spring_model.json")
    angles = np.deg2rad(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles, relax_internal=True)
    basis = interpolate_basis(basis_by_angle, angles, theta)
    schedules = make_schedules(
        basis,
        target,
        args.cycles,
        args.samples_per_cycle,
        1.0,
        800.0,
        args.update_fraction,
    )

    results = {}
    predictions = {}
    for mode in ("fixed", "controlled", "per_cycle_upper_bound"):
        schedule = schedules[mode]
        results[mode] = evaluate_schedule(t, theta, target, basis, schedule)
        predictions[mode] = np.sum(basis * schedule, axis=1)

    score = periodicity_score(t, theta, target, nominal_frequency_hz=1.0)
    period = 1.0
    update_times = np.arange(args.cycles, dtype=float) * period
    controlled = schedules["controlled"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, target, color="black", linewidth=1.2, label="target torque")
    axes[0].plot(t, predictions["fixed"], linewidth=1.0, label="fixed fan")
    axes[0].plot(t, predictions["controlled"], linewidth=1.1, label="period-limited fan")
    axes[0].set_ylabel("Torque [Nm]")
    axes[0].legend(ncol=3)

    axes[1].plot(t, np.mean(controlled, axis=1), color="tab:green", drawstyle="steps-post")
    axes[1].set_ylabel("Mean stiffness [N/m]")
    axes[1].set_title("Stiffness is constant inside each gait period")

    for spring_index in range(controlled.shape[1]):
        axes[2].plot(t, controlled[:, spring_index], linewidth=0.7, alpha=0.65)
    axes[2].set_ylabel("Individual stiffness [N/m]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        for boundary in update_times:
            ax.axvline(boundary, color="tab:red", linestyle="--", linewidth=0.65, alpha=0.55)
        ax.grid(True, alpha=0.22)
    fig.suptitle(
        f"Period-limited adaptation: T={period:.1f}s, {args.cycles} allowed updates, "
        f"periodicity score={score['periodicity_score']:.3f}"
    )
    fig.tight_layout()
    output = PROJECT_ROOT / "plots" / "legacy" / "dataset_examples" / f"{args.output_name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    possible_timestep_updates = len(t)
    print(f"Nominal period: {period:.3f} s")
    print(f"Timesteps per period: {args.samples_per_cycle}")
    print(f"Allowed stiffness updates: {args.cycles}")
    print(f"Possible per-timestep updates: {possible_timestep_updates}")
    print(f"Update reduction: {100.0 * (1.0 - args.cycles / possible_timestep_updates):.2f}%")
    print(f"Periodicity score: {score['periodicity_score']:.4f}")
    for mode, values in results.items():
        print(f"{mode:21s} | RMSE {values[0]:.3f} Nm | power offload {values[2]:.2f}%")
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
