#!/usr/bin/env python3
"""Compare profile blending with actuator-tuned PEJ stiffness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from examples.common.reporting import print_section, print_table, write_csv

import pej


K_SOFT = 4.0
K_STIFF = 8.0
PHI_MIN = -0.35
PHI_MAX = 0.35


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
    table_path = write_csv("artifacts/tables/actuator_tuned_pej.csv", rows)

    print("Actuator-tuned adaptive PEJ prototype")
    print(f"  k_soft / k_stiff:        {K_SOFT:.2f} / {K_STIFF:.2f} N m/rad")
    print(f"  phi_min / phi_max:       {PHI_MIN:.2f} / {PHI_MAX:.2f} rad")
    print(f"  Flat roughness calib:    {min_score:.4f}")
    print(f"  Rough roughness calib:   {max_score:.4f}")
    print_section("Energy comparison")
    print_table(
        rows,
        [
            "scenario",
            "case",
            "motor_energy_j",
            "mean_power_w",
            "net_saved_j",
            "offload_pct",
            "mean_q",
            "mean_phi",
            "mean_k",
            "spring_k",
        ],
    )
    print(f"\n  Wrote table: {table_path}")


def synthetic_terrain(kind: str, *, duration: float = 8.0, dt: float = 0.02) -> pej.TrajectoryData:
    time = np.arange(0.0, duration + dt, dt)
    if kind == "flat":
        theta = 0.22 * np.sin(2.0 * np.pi * 1.6 * time)
        stiffness = K_SOFT
        cubic_stiffness = 2.0
        disturbance = 0.18 * np.sin(2.0 * np.pi * 3.2 * time + 0.2)
    elif kind == "rough":
        theta = (
            0.25 * np.sin(2.0 * np.pi * 1.6 * time)
            + 0.06 * np.sin(2.0 * np.pi * 5.2 * time + 0.4)
        )
        stiffness = K_STIFF
        cubic_stiffness = 4.0
        disturbance = 0.36 * np.sin(2.0 * np.pi * 6.4 * time + 0.7)
    else:
        raise ValueError(f"unsupported terrain kind: {kind}")

    theta_dot = np.gradient(theta, dt)
    tau_elastic = stiffness * theta + cubic_stiffness * theta**3
    tau_total = tau_elastic + disturbance
    return pej.TrajectoryData(
        time=time,
        joint_name=np.full(time.shape, "front_thigh"),
        theta=theta,
        theta_dot=theta_dot,
        tau_total=tau_total,
        terrain=np.full(time.shape, kind),
        policy=np.full(time.shape, "actuator_tuned_demo"),
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
    scores = rolling_roughness(trajectory.theta_dot)
    q = pej.roughness_to_q(scores, min_score, max_score)
    blended_tau = pej.blend_profiles(trajectory.theta, flat_profile, rough_profile, q)
    actuator = pej.actuator_tuned_stiffness(
        trajectory.theta,
        q,
        k_soft=K_SOFT,
        k_stiff=K_STIFF,
        phi_min=PHI_MIN,
        phi_max=PHI_MAX,
    )
    cases = [
        ("fixed soft spring", K_SOFT * trajectory.theta, 0.0, PHI_MIN, K_SOFT, f"{K_SOFT:.2f}"),
        ("fixed stiff spring", K_STIFF * trajectory.theta, 1.0, PHI_MAX, K_STIFF, f"{K_STIFF:.2f}"),
        ("old profile blend", blended_tau, float(np.mean(q)), np.nan, np.nan, "profile blend"),
        (
            "actuator tuned",
            actuator.tau_spring,
            float(np.mean(actuator.q)),
            float(np.mean(actuator.phi)),
            float(np.mean(actuator.k_eff)),
            f"{K_SOFT:.2f}-{K_STIFF:.2f}",
        ),
    ]

    terrain_name = "/".join(sorted(set(trajectory.terrain)))
    baseline_power = pej.motor_power(trajectory.tau_total, trajectory.theta_dot)
    baseline_energy = float(np.trapezoid(baseline_power, trajectory.time))
    power_without = float(np.mean(baseline_power))
    rows = []
    for case_name, tau_pej, mean_q, mean_phi, mean_k, spring_k in cases:
        motor_power_with = pej.residual_motor_power(trajectory.tau_total, tau_pej, trajectory.theta_dot)
        power_with = float(np.mean(motor_power_with))
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
                "mean_q": mean_q,
                "mean_phi": "" if np.isnan(mean_phi) else mean_phi,
                "mean_k": "" if np.isnan(mean_k) else mean_k,
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
