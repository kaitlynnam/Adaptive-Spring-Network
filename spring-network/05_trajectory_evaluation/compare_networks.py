from pathlib import Path
import argparse
import csv
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))
sys.path.insert(0, str(PROJECT_ROOT / "05_trajectory_evaluation"))

from evaluate_trajectory import (
    DEFAULT_BATCH_COUNT,
    DEFAULT_BATCH_SEED,
    average_metrics,
    evaluate_arrays,
    _format_cell,
    _format_table,
)
from train_adaptive_dataset import generate_motion_trajectory, generate_profile_parameters
from topology_loader import DEFAULT_TOPOLOGY_PATH


DEFAULT_ADAPTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "adaptive_trained_model.npz"


def run_model_batch(args, model_name, adaptive_model_path, profile_params):
    model_args = argparse.Namespace(
        topology=args.topology,
        adaptive_model=adaptive_model_path,
        no_relax_internal=args.no_relax_internal,
    )
    rows = []
    for index, params in enumerate(profile_params):
        t, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params,
            duration=args.duration,
            samples=args.samples,
            seed=args.batch_seed + index,
        )
        result = evaluate_arrays(
            model_args,
            profile=params["name"],
            t=t,
            theta=theta,
            theta_dot=theta_dot,
            theta_ddot=theta_ddot,
            tau_target=tau_target,
            duration=args.duration,
            samples=args.samples,
            amplitude_deg=params["amplitude_deg"],
            frequency_hz=params["frequency_hz"],
            profile_params=params,
        )
        result["model"] = model_name
        result["family"] = params["family"]
        rows.append(result)
    return rows


def summary_row(group, model_name, rows):
    average = average_metrics(rows)
    return {
        "group": group,
        "model": model_name,
        "cases": len(rows),
        "average_baseline_motor_energy_j": average["baseline_motor_energy_j"],
        "average_motor_energy_with_spring_j": average["motor_energy_with_spring_j"],
        "average_energy_saved_j": average["energy_saved_j"],
        "average_offload_pct": average["offload_pct"],
        "average_mean_abs_torque_error_nm": average["mean_abs_torque_error_nm"],
        "average_max_abs_torque_error_nm": average["max_abs_torque_error_nm"],
    }


def build_summary_rows(model_results):
    summary_rows = []
    for model_name, rows in model_results.items():
        summary_rows.append(summary_row("overall", model_name, rows))
        for family in sorted({row["family"] for row in rows}):
            family_rows = [row for row in rows if row["family"] == family]
            summary_rows.append(summary_row(family, model_name, family_rows))
    return summary_rows


def print_summary(summary_rows):
    print("Trajectory model comparison")
    print("---------------------------")
    print(_format_table(summary_rows, ["group", "model", "cases", "average_offload_pct", "average_energy_saved_j", "average_mean_abs_torque_error_nm"]))


def write_summary_csv(path, summary_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "group",
        "model",
        "cases",
        "average_baseline_motor_energy_j",
        "average_motor_energy_with_spring_j",
        "average_energy_saved_j",
        "average_offload_pct",
        "average_mean_abs_torque_error_nm",
        "average_max_abs_torque_error_nm",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: _format_cell(row[column]) for column in columns})


def run(args):
    rng = np.random.default_rng(args.batch_seed)
    profile_params = generate_profile_parameters(rng, args.batch_count)
    model_results = {
        "baseline_model": run_model_batch(args, "baseline_model", None, profile_params),
        "adaptive_trained_model": run_model_batch(args, "adaptive_trained_model", args.adaptive_model, profile_params),
    }
    summary_rows = build_summary_rows(model_results)
    print_summary(summary_rows)

    output_path = PROJECT_ROOT / "tables" / "trajectory_model_comparison.csv"
    write_summary_csv(output_path, summary_rows)
    print()
    print(f"Saved trajectory model comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and adaptive trained models on real generated trajectories.")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_PATH, help="Baseline topology JSON file.")
    parser.add_argument("--adaptive-model", default=DEFAULT_ADAPTIVE_MODEL_PATH, help="Adaptive trained model .npz file.")
    parser.add_argument("--duration", type=float, default=5.0, help="Generated trajectory duration in seconds.")
    parser.add_argument("--samples", type=int, default=300, help="Samples per generated trajectory.")
    parser.add_argument("--batch-count", type=int, default=DEFAULT_BATCH_COUNT, help="Generated trajectory/profile count.")
    parser.add_argument("--batch-seed", type=int, default=DEFAULT_BATCH_SEED, help="Random seed for generated profiles.")
    parser.add_argument(
        "--no-relax-internal",
        action="store_true",
        help="Disable quasi-static internal-node relaxation for the baseline model.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
