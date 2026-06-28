#!/usr/bin/env python3
"""Distill a passive PEJ profile from trajectory samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    power_without = float(np.mean(pej.motor_power(joint.tau_total, joint.theta_dot)))
    power_with = float(np.mean(pej.residual_motor_power(joint.tau_total, tau_pej, joint.theta_dot)))
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


if __name__ == "__main__":
    main()
