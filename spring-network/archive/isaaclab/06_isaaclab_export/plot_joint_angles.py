"""Plot all joint angles from an exported Isaac Lab rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path, help="Exported rollout .npz file")
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.rollout) as data:
        time = np.asarray(data["time"], dtype=float)
        angles = np.rad2deg(np.asarray(data["theta"], dtype=float))
        names = [str(name) for name in data["joint_names"]]

    if not 0 <= args.env_id < angles.shape[1]:
        raise ValueError(f"--env-id must be between 0 and {angles.shape[1] - 1}")

    output = args.output or args.rollout.with_name(f"{args.rollout.stem}_joint_angles_env{args.env_id}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
    for joint_index, (axis, name) in enumerate(zip(axes.flat, names)):
        axis.plot(time[:, args.env_id], angles[:, args.env_id, joint_index], linewidth=1.8)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        axis.set_title(name.replace("_joint", ""))
        axis.grid(alpha=0.25)

    for axis in axes[:, 0]:
        axis.set_ylabel("angle [deg]")
    for axis in axes[-1, :]:
        axis.set_xlabel("time [s]")

    fig.suptitle(f"Go2 joint angles — rollout environment {args.env_id}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output.resolve())


if __name__ == "__main__":
    main()
