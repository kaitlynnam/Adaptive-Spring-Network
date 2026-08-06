"""Plot joint torque versus angle from an exported Isaac Lab rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument(
        "--torque-source",
        choices=("tau_total", "tau_applied", "tau_computed"),
        default="tau_total",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.rollout) as data:
        time = np.asarray(data["time"], dtype=float)[:, args.env_id]
        angle = np.rad2deg(np.asarray(data["theta"], dtype=float)[:, args.env_id, :])
        torque = np.asarray(data[args.torque_source], dtype=float)[:, args.env_id, :]
        names = [str(name) for name in data["joint_names"]]

    output = args.output or args.rollout.with_name(
        f"{args.rollout.stem}_torque_vs_angle_env{args.env_id}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    scatter = None
    for joint_index, (axis, name) in enumerate(zip(axes.flat, names)):
        axis.plot(angle[:, joint_index], torque[:, joint_index], color="0.72", linewidth=0.8)
        scatter = axis.scatter(
            angle[:, joint_index],
            torque[:, joint_index],
            c=time,
            cmap="viridis",
            s=18,
            edgecolors="none",
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        axis.set_title(name.replace("_joint", ""))
        axis.set_xlabel("angle [deg]")
        axis.set_ylabel("torque [N·m]")
        axis.grid(alpha=0.25)

    fig.suptitle(f"Go2 torque versus joint angle — rollout environment {args.env_id}", fontsize=16)
    fig.subplots_adjust(right=0.9, top=0.91, hspace=0.35, wspace=0.3)
    color_axis = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(scatter, cax=color_axis, label="time [s]")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output.resolve())


if __name__ == "__main__":
    main()
