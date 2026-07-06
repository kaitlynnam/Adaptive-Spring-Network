#!/usr/bin/env python3
"""Exercise the reproduced PEJ paper equations in Python."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import pej


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]

    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    tables = pej.paper_tables()

    co_design_offload = pej.offload_percentage(
        tables.power.co_design.before, tables.power.co_design.after
    )
    reference_offload = pej.offload_percentage(
        tables.power.reference.before, tables.power.reference.after
    )

    print("Paper Table 4 offload reproduction:")
    print_table(
        ["Terrain", "CoDesignOffloadPct", "ReferenceOffloadPct"],
        [
            [terrain, f"{co_design:.1f}", f"{reference:.1f}"]
            for terrain, co_design, reference in zip(
                tables.terrain_labels, co_design_offload, reference_offload
            )
        ],
    )

    dt = 0.02
    t = np.arange(0.0, 20.0 + dt, dt)
    theta = 0.35 + 0.25 * np.sin(2.0 * np.pi * 1.8 * t)
    theta_dot = np.gradient(theta, dt)
    tau_elastic_true = 4.0 * (theta - 0.35) + 9.0 * (theta - 0.35) ** 3
    tau_total = tau_elastic_true + 0.4 * np.sin(2.0 * np.pi * 3.6 * t + 0.2)

    profile, history = pej.distill_profile(
        theta,
        theta_dot,
        tau_total,
        num_knots=20,
        max_iterations=300,
        learning_rate=1e-4,
        momentum=0.8,
    )
    tau_pej = pej.eval_piecewise_profile(theta, profile.theta, profile.tau)
    power_without = np.mean(pej.motor_power(tau_total, theta_dot))
    power_with = np.mean(pej.residual_motor_power(tau_total, tau_pej, theta_dot))

    print("\nSynthetic PEJ distillation:")
    print(f"  Mean power without PEJ: {power_without:.4f} W")
    print(f"  Mean power with PEJ:    {power_with:.4f} W")
    print(f"  Offload:                {pej.offload_percentage(power_without, power_with):.2f} %")
    print(f"  Final objective:        {history.objective[-1]:.4f}")

    v_actual = np.column_stack((np.ones_like(t) * 0.8, np.zeros_like(t), np.zeros_like(t)))
    v_cmd_unit = np.array([1.0, 0.0, 0.0])
    v_scalar = pej.sliding_mean(pej.projected_speed(v_actual, v_cmd_unit), 10)
    joint_power = np.column_stack(
        (pej.motor_power(tau_total, theta_dot), pej.motor_power(0.5 * tau_total, theta_dot))
    )
    cot = pej.cost_of_transport(joint_power, v_scalar)
    r_total = pej.total_reward(1.0, 0.5, cot)
    ev = pej.tracking_error(t, np.ones_like(t) * 0.9, v_actual[:, 0])

    print("\nOther equations:")
    print(f"  Mean CoT:               {np.mean(cot):.4f}")
    print(f"  Mean total reward:      {np.mean(r_total):.4f}")
    print(f"  Tracking error:         {ev:.4f} m/s")

    cam = pej.cam_radius_from_torque(
        profile.active_theta,
        profile.active_tau,
        spring_rate=14.9e3,
        base_radius=0.050,
        preload_deflection=0.0,
    )
    print("\nSynthetic cam mapping:")
    print(f"  Energy range:           {np.max(cam.energy) - np.min(cam.energy):.4f} J")
    print(f"  Radius range:           {np.min(cam.radius) * 1000:.2f} to {np.max(cam.radius) * 1000:.2f} mm")


if __name__ == "__main__":
    main()
