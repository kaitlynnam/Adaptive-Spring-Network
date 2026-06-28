import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pej


def test_load_trajectory_csv_and_filter_joint(tmp_path):
    path = tmp_path / "trajectory.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "joint_name", "theta", "theta_dot", "tau_total", "terrain"])
        writer.writerow([0.00, "front_thigh", 0.1, 1.0, 2.0, "flat"])
        writer.writerow([0.02, "rear_thigh", 0.2, 1.5, 3.0, "flat"])
        writer.writerow([0.04, "front_thigh", 0.3, 2.0, 4.0, "flat"])

    trajectory = pej.load_trajectory(path)
    front_thigh = trajectory.for_joint("front_thigh")

    assert trajectory.joint_names == ["front_thigh", "rear_thigh"]
    np.testing.assert_allclose(front_thigh.theta, np.array([0.1, 0.3]))
    np.testing.assert_allclose(front_thigh.tau_total, np.array([2.0, 4.0]))
    np.testing.assert_array_equal(front_thigh.terrain, np.array(["flat", "flat"]))


def test_synthetic_trajectory_can_be_distilled_to_offload_power():
    trajectory = pej.synthetic_trajectory(duration=2.0)
    trajectory.for_joint("front_thigh")

    profile, _history = pej.distill_profile(
        trajectory.theta,
        trajectory.theta_dot,
        trajectory.tau_total,
        num_knots=10,
        max_iterations=50,
        learning_rate=1e-4,
        torque_limit=23.5,
    )
    tau_pej = pej.eval_piecewise_profile(trajectory.theta, profile.theta, profile.tau)
    power_without = np.mean(pej.motor_power(trajectory.tau_total, trajectory.theta_dot))
    power_with = np.mean(pej.residual_motor_power(trajectory.tau_total, tau_pej, trajectory.theta_dot))

    assert power_with < power_without
