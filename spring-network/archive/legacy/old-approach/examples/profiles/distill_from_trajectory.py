#!/usr/bin/env python3
"""Distill a passive PEJ profile from trajectory samples."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import pej


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV or NPZ trajectory file. Uses a deterministic synthetic rollout if omitted.",
    )
    parser.add_argument("--joint", default="front_thigh", help="joint_name to distill")
    parser.add_argument("--num-knots", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.8)
    parser.add_argument("--torque-limit", type=float, default=23.5)
    parser.add_argument("--spring-rate", type=float, default=14.9e3)
    parser.add_argument("--base-radius", type=float, default=0.050)
    parser.add_argument("--preload-deflection", type=float, default=0.0)
    parser.add_argument("--output-profile", type=Path, help="optional CSV output path")
    parser.add_argument(
        "--output-plots",
        type=Path,
        default=Path("artifacts/plots/profile_distillation"),
        help="directory for generated PNG plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = pej.load_trajectory(args.input) if args.input else pej.synthetic_trajectory()
    joint = trajectory.for_joint(args.joint)

    profile, history = pej.distill_profile(
        joint.theta,
        joint.theta_dot,
        joint.tau_total,
        num_knots=args.num_knots,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        torque_limit=args.torque_limit,
    )

    tau_pej = pej.eval_piecewise_profile(joint.theta, profile.theta, profile.tau)
    residual_tau = pej.residual_torque(joint.tau_total, tau_pej)
    motor_power_without = pej.motor_power(joint.tau_total, joint.theta_dot)
    motor_power_with = pej.residual_motor_power(joint.tau_total, tau_pej, joint.theta_dot)
    power_without = float(np.mean(pej.motor_power(joint.tau_total, joint.theta_dot)))
    power_with = float(np.mean(motor_power_with))
    offload = float(pej.offload_percentage(power_without, power_with))

    cam = pej.cam_radius_from_torque(
        profile.active_theta,
        profile.active_tau,
        spring_rate=args.spring_rate,
        base_radius=args.base_radius,
        preload_deflection=args.preload_deflection,
    )

    print("PEJ trajectory distillation")
    print(f"  Source:                 {args.input if args.input else 'synthetic fixture'}")
    print(f"  Joint:                  {args.joint}")
    print(f"  Samples:                {joint.time.size}")
    print(f"  Angle range:            {profile.active_theta[0]:.4f} to {profile.active_theta[-1]:.4f} rad")
    print(f"  Mean power without PEJ: {power_without:.4f} W")
    print(f"  Mean power with PEJ:    {power_with:.4f} W")
    print(f"  Offload:                {offload:.2f} %")
    print(f"  Final objective:        {history.objective[-1]:.4f}")
    print(f"  Final grad norm:        {history.gradient_norm[-1]:.4f}")
    print(f"  Cam radius range:       {np.min(cam.radius) * 1000:.2f} to {np.max(cam.radius) * 1000:.2f} mm")

    if args.output_profile:
        write_profile(args.output_profile, profile, cam)
        print(f"  Wrote profile:          {args.output_profile}")

    plot_paths = write_plots(
        args.output_plots,
        joint,
        profile,
        cam,
        tau_pej,
        residual_tau,
        motor_power_without,
        motor_power_with,
    )
    for path in plot_paths:
        print(f"  Wrote plot:             {path}")


def write_profile(path: Path, profile: pej.PiecewiseProfile, cam: pej.CamProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_rad", "tau_nm", "cam_radius_m", "spring_deflection_m", "energy_j"])
        writer.writerows(
            zip(
                profile.active_theta,
                profile.active_tau,
                cam.radius,
                cam.spring_deflection,
                cam.energy,
            )
        )


def write_plots(
    output_dir: Path,
    joint: pej.TrajectoryData,
    profile: pej.PiecewiseProfile,
    cam: pej.CamProfile,
    tau_pej: np.ndarray,
    residual_tau: np.ndarray,
    motor_power_without: np.ndarray,
    motor_power_with: np.ndarray,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paths.append(
        _save_plot(
            output_dir / "total_torque_vs_angle.png",
            "Total torque vs angle",
            "Joint angle theta (rad)",
            "Total torque (N m)",
            lambda ax: ax.scatter(joint.theta, joint.tau_total, s=8, alpha=0.35),
        )
    )
    paths.append(
        _save_plot(
            output_dir / "learned_pej_torque_vs_angle.png",
            "Learned PEJ torque vs angle",
            "Joint angle theta (rad)",
            "PEJ torque (N m)",
            lambda ax: (
                ax.plot(profile.theta, profile.tau, linewidth=2.0),
                ax.scatter(joint.theta, tau_pej, s=8, alpha=0.25),
            ),
        )
    )
    paths.append(
        _save_plot(
            output_dir / "residual_torque_vs_time.png",
            "Residual torque vs time",
            "Time (s)",
            "Residual torque (N m)",
            lambda ax: ax.plot(joint.time, residual_tau, linewidth=1.2),
        )
    )
    paths.append(
        _save_plot(
            output_dir / "motor_power_before_after_pej.png",
            "Motor power before/after PEJ",
            "Time (s)",
            "Positive motor power (W)",
            lambda ax: (
                ax.plot(joint.time, motor_power_without, label="Before PEJ", linewidth=1.2),
                ax.plot(joint.time, motor_power_with, label="After PEJ", linewidth=1.2),
                ax.legend(),
            ),
        )
    )
    paths.append(
        _save_plot(
            output_dir / "cam_radius_vs_angle.png",
            "Cam radius vs angle",
            "Joint angle theta (rad)",
            "Cam radius (mm)",
            lambda ax: ax.plot(cam.theta, cam.radius * 1000.0, linewidth=2.0),
        )
    )
    return paths


def _save_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    draw: Callable[[plt.Axes], object],
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    draw(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
