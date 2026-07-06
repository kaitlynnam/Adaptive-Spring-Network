#!/usr/bin/env python3
"""Compare fixed and adaptive PEJ profiles on synthetic terrains."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from examples.common.reporting import print_section, print_table, write_csv

import pej


def main() -> None:
    flat = synthetic_terrain("flat")
    rough = synthetic_terrain("rough")
    mixed = concatenate_trajectories(flat, rough)

    flat_profile, _ = pej.distill_profile(
        flat.theta,
        flat.theta_dot,
        flat.tau_total,
        num_knots=20,
        max_iterations=300,
        learning_rate=1e-4,
        momentum=0.8,
        torque_limit=23.5,
    )
    rough_profile, _ = pej.distill_profile(
        rough.theta,
        rough.theta_dot,
        rough.tau_total,
        num_knots=20,
        max_iterations=300,
        learning_rate=1e-4,
        momentum=0.8,
        torque_limit=23.5,
    )

    flat_scores = rolling_roughness(flat.theta_dot)
    rough_scores = rolling_roughness(rough.theta_dot)
    min_score = float(np.percentile(flat_scores, 25.0))
    max_score = float(np.percentile(rough_scores, 75.0))

    rows = (
        evaluate_all(flat, flat_profile, rough_profile, min_score, max_score)
        + evaluate_all(rough, flat_profile, rough_profile, min_score, max_score)
        + evaluate_all(mixed, flat_profile, rough_profile, min_score, max_score)
    )
    table_path = write_csv("artifacts/tables/adaptive_spring_network.csv", rows)

    print("Adaptive PEJ spring-network prototype")
    print(f"  Flat roughness calibration:  {min_score:.4f}")
    print(f"  Rough roughness calibration: {max_score:.4f}")
    print_section("Energy comparison")
    print_table(
        rows,
        ["scenario", "case", "motor_energy_j", "mean_power_w", "net_saved_j", "offload_pct", "mean_q", "spring_k"],
    )
    print(f"\n  Wrote table: {table_path}")


def synthetic_terrain(kind: str, *, duration: float = 8.0, dt: float = 0.02) -> pej.TrajectoryData:
    time = np.arange(0.0, duration + dt, dt)
    center = 0.35
    if kind == "flat":
        theta = center + 0.22 * np.sin(2.0 * np.pi * 1.6 * time)
        stiffness = 4.0
        cubic_stiffness = 9.0
        disturbance = 0.20 * np.sin(2.0 * np.pi * 3.2 * time + 0.2)
    elif kind == "rough":
        theta = (
            center
            + 0.25 * np.sin(2.0 * np.pi * 1.6 * time)
            + 0.06 * np.sin(2.0 * np.pi * 5.2 * time + 0.4)
        )
        stiffness = 7.0
        cubic_stiffness = 18.0
        disturbance = 0.45 * np.sin(2.0 * np.pi * 6.4 * time + 0.7)
    else:
        raise ValueError(f"unsupported terrain kind: {kind}")

    theta_dot = np.gradient(theta, dt)
    tau_elastic = stiffness * (theta - center) + cubic_stiffness * (theta - center) ** 3
    tau_total = tau_elastic + disturbance
    return pej.TrajectoryData(
        time=time,
        joint_name=np.full(time.shape, "front_thigh"),
        theta=theta,
        theta_dot=theta_dot,
        tau_total=tau_total,
        terrain=np.full(time.shape, kind),
        policy=np.full(time.shape, "adaptive_demo"),
        robot_id=np.full(time.shape, "0"),
    )


def concatenate_trajectories(first: pej.TrajectoryData, second: pej.TrajectoryData) -> pej.TrajectoryData:
    dt = float(np.median(np.diff(first.time)))
    second_time = second.time + first.time[-1] + dt
    return pej.TrajectoryData(
        time=np.concatenate((first.time, second_time)),
        joint_name=np.concatenate((first.joint_name, second.joint_name)),
        theta=np.concatenate((first.theta, second.theta)),
        theta_dot=np.concatenate((first.theta_dot, second.theta_dot)),
        tau_total=np.concatenate((first.tau_total, second.tau_total)),
        terrain=np.concatenate((first.terrain, second.terrain)),
        policy=np.concatenate((first.policy, second.policy)),
        robot_id=np.concatenate((first.robot_id, second.robot_id)),
    )


def evaluate_all(
    trajectory: pej.TrajectoryData,
    flat_profile: pej.PiecewiseProfile,
    rough_profile: pej.PiecewiseProfile,
    min_score: float,
    max_score: float,
) -> list[dict[str, object]]:
    baseline_power = pej.motor_power(trajectory.tau_total, trajectory.theta_dot)
    baseline_energy = float(np.trapezoid(baseline_power, trajectory.time))
    power_without = float(np.mean(baseline_power))
    scores = rolling_roughness(trajectory.theta_dot)
    q = pej.roughness_to_q(scores, min_score, max_score)
    adaptive_tau = pej.blend_profiles(trajectory.theta, flat_profile, rough_profile, q)
    cases = [
        ("no spring", np.zeros_like(trajectory.theta), np.nan, ""),
        ("fixed flat", pej.eval_piecewise_profile(trajectory.theta, flat_profile.theta, flat_profile.tau), 0.0, "profile"),
        ("fixed rough", pej.eval_piecewise_profile(trajectory.theta, rough_profile.theta, rough_profile.tau), 1.0, "profile"),
        ("adaptive blend", adaptive_tau, float(np.mean(q)), "profile blend"),
    ]

    terrain_name = "mixed" if trajectory.terrain is None else "/".join(sorted(set(trajectory.terrain)))
    rows = []
    for case_name, tau_pej, mean_q, spring_k in cases:
        power_with = float(np.mean(pej.residual_motor_power(trajectory.tau_total, tau_pej, trajectory.theta_dot)))
        motor_power_with = pej.residual_motor_power(trajectory.tau_total, tau_pej, trajectory.theta_dot)
        motor_energy = float(np.trapezoid(motor_power_with, trajectory.time))
        offload = float(pej.offload_percentage(power_without, power_with))
        rows.append(
            {
                "scenario": terrain_name,
                "case": case_name,
                "motor_energy_j": motor_energy,
                "mean_power_w": power_with,
                "net_saved_j": baseline_energy - motor_energy,
                "offload_pct": offload,
                "mean_q": "" if np.isnan(mean_q) else mean_q,
                "spring_k": spring_k,
            }
        )
    return rows


def rolling_roughness(theta_dot: np.ndarray, window_length: int = 25) -> np.ndarray:
    scores = np.empty_like(theta_dot, dtype=float)
    for i in range(theta_dot.size):
        start = max(0, i + 1 - window_length)
        scores[i] = pej.roughness_score(theta_dot[start : i + 1])
    return scores


if __name__ == "__main__":
    main()
