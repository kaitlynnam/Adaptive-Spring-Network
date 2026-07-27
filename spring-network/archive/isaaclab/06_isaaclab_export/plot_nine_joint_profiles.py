"""Plot torque-versus-angle profiles for one joint in nine rollout environments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--joint", default="FL_calf_joint")
    parser.add_argument("--terrain-family", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with np.load(args.rollout) as data:
        names = np.asarray(data["joint_names"], dtype=str)
        matches = np.flatnonzero(names == args.joint)
        if not len(matches):
            raise ValueError(f"Unknown joint {args.joint!r}")
        joint_index = int(matches[0])
        terrain_all = np.asarray(data["terrain_family"], dtype=str) if "terrain_family" in data.files else None
        env_ids = np.arange(data["theta"].shape[1])
        if args.terrain_family is not None:
            if terrain_all is None:
                raise ValueError("This rollout has no terrain_family labels.")
            env_ids = env_ids[terrain_all == args.terrain_family]
        env_ids = env_ids[:9]
        angle = np.rad2deg(np.asarray(data["theta"][:, env_ids, joint_index], dtype=float))
        torque = np.asarray(data["tau_total"][:, env_ids, joint_index], dtype=float)
        time = np.asarray(data["time"][:, env_ids], dtype=float)
        terrain = terrain_all[env_ids] if terrain_all is not None else None
    if angle.shape[1] < 9:
        raise ValueError(f"The rollout contains only {angle.shape[1]} environments; nine are required.")

    output = args.output or args.rollout.with_name(f"{args.rollout.stem}_{args.joint}_nine_profiles.png")
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    scatter = None
    for plot_index, axis in enumerate(axes.flat):
        axis.plot(angle[:, plot_index], torque[:, plot_index], color="0.75", linewidth=0.7)
        scatter = axis.scatter(angle[:, plot_index], torque[:, plot_index], c=time[:, plot_index], cmap="viridis", s=10)
        family = f" - {terrain[plot_index]}" if terrain is not None else ""
        axis.set_title(f"environment {env_ids[plot_index]}{family}")
        axis.set_xlabel("angle [deg]")
        axis.set_ylabel("torque [N*m]")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Nine {args.joint} torque-angle profiles", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    color_axis = fig.add_axes([0.95, 0.13, 0.015, 0.74])
    fig.colorbar(scatter, cax=color_axis, label="time [s]")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output.resolve())


if __name__ == "__main__":
    main()
