"""Extract one joint trajectory from an Isaac Lab rollout NPZ.

This creates the CSV shape expected by:

    python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --trajectory <csv>

The output columns are:

    t, theta, theta_dot, tau_target, profile
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "isaaclab_rollouts" / "single_joint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one env/joint CSV from an exported Isaac Lab rollout NPZ.")
    parser.add_argument("input_npz", help="NPZ produced by export_pretrained_rollout.py.")
    parser.add_argument("--joint", required=True, help="Joint name or integer joint index to extract.")
    parser.add_argument("--env-id", type=int, default=0, help="Environment index to extract.")
    parser.add_argument(
        "--torque-source",
        choices=("tau_total", "tau_applied", "tau_computed"),
        default="tau_total",
        help="Torque array to write as tau_target.",
    )
    parser.add_argument("--profile", default="isaaclab_rollout", help="Profile label stored in the CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the extracted CSV.")
    parser.add_argument("--output-name", default=None, help="Optional output filename.")
    return parser.parse_args()


def resolve_joint_index(joint_arg: str, joint_names: np.ndarray) -> int:
    try:
        index = int(joint_arg)
    except ValueError:
        matches = np.where(joint_names.astype(str) == joint_arg)[0]
        if matches.size == 0:
            available = ", ".join(joint_names.astype(str).tolist())
            raise ValueError(f"Joint {joint_arg!r} was not found. Available joints: {available}")
        index = int(matches[0])
    if index < 0 or index >= len(joint_names):
        raise IndexError(f"Joint index {index} is out of range 0..{len(joint_names) - 1}.")
    return index


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_npz)
    with np.load(input_path, allow_pickle=False) as data:
        joint_names = np.asarray(data["joint_names"], dtype=str)
        joint_index = resolve_joint_index(args.joint, joint_names)
        time = np.asarray(data["time"], dtype=float)[:, args.env_id]
        theta = np.asarray(data["theta"], dtype=float)[:, args.env_id, joint_index]
        theta_dot = np.asarray(data["theta_dot"], dtype=float)[:, args.env_id, joint_index]
        tau_target = np.asarray(data[args.torque_source], dtype=float)[:, args.env_id, joint_index]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        safe_joint = str(joint_names[joint_index]).replace("/", "_").replace("\\", "_").replace(":", "_")
        output_name = f"{input_path.stem}_env{args.env_id}_{safe_joint}.csv"
    output_path = output_dir / output_name

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["t", "theta", "theta_dot", "tau_target", "profile"])
        for row in zip(time, theta, theta_dot, tau_target):
            writer.writerow([f"{float(row[0]):.9g}", f"{float(row[1]):.9g}", f"{float(row[2]):.9g}", f"{float(row[3]):.9g}", args.profile])

    print(f"Extracted joint: {joint_names[joint_index]} (index {joint_index})")
    print(f"Samples: {len(time)}")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
