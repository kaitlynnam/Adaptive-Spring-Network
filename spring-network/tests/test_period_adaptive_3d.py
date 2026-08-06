import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_adaptive_learning"))

from train_period_adaptive_3d import period_observation, rollout


def test_period_observation_contains_all_six_channels_in_time_order():
    values = [torch.tensor([[1.0, 2.0]]) * factor for factor in range(1, 7)]
    result = period_observation(*values, torch.ones(3), 1.0)
    assert result.shape == (1, 12)
    np.testing.assert_allclose(result.numpy().reshape(1, 2, 6)[0, 0], [1, 2, 3, 4, 5, 6])


def test_first_period_is_default_and_later_stiffness_is_held():
    dataset = {
        "basis": torch.ones((1, 3, 2)), "target": torch.zeros((1, 3)),
        "theta": torch.zeros((1, 3)), "theta_dot": torch.zeros((1, 3)),
        "theta_ddot": torch.zeros((1, 3)), "motion_scales": torch.ones(3),
        "torque_scale": 1.0,
    }
    parameters = {
        "w1": torch.zeros((18, 1)), "b1": torch.zeros(1),
        "w2": torch.zeros((1, 2)), "b2": torch.zeros(2),
    }
    torque, stiffness = rollout(parameters, dataset, torch.tensor([2.0, 3.0]), 0.0, 3)
    np.testing.assert_allclose(stiffness[0, 0].numpy(), [2.0, 3.0])
    np.testing.assert_allclose(torque[0, 0].numpy(), [5.0, 5.0, 5.0])
    np.testing.assert_allclose(stiffness[0, 1].numpy(), np.log(2.0) * np.ones(2))
    # There is one vector per period, with no timestep-level stiffness axis.
    assert stiffness.shape == (1, 3, 2)
    np.testing.assert_allclose(torque[0, 1].numpy(), torque[0, 1, 0].numpy())


def test_rollout_accepts_randomized_initial_stiffness_per_trajectory():
    dataset = {
        "basis": torch.ones((2, 2, 1)), "target": torch.zeros((2, 2)),
        "theta": torch.zeros((2, 2)), "theta_dot": torch.zeros((2, 2)),
        "theta_ddot": torch.zeros((2, 2)), "motion_scales": torch.ones(3),
        "torque_scale": 1.0,
    }
    parameters = {
        "w1": torch.zeros((12, 1)), "b1": torch.zeros(1),
        "w2": torch.zeros((1, 1)), "b2": torch.zeros(1),
    }
    initial = torch.tensor([[2.0], [5.0]])
    torque, stiffness = rollout(parameters, dataset, initial, 0.0, 2)
    np.testing.assert_allclose(stiffness[:, 0].numpy(), initial.numpy())
    np.testing.assert_allclose(torque[:, 0].numpy(), [[2.0, 2.0], [5.0, 5.0]])


def test_rollout_uses_a_distinct_refreshed_basis_for_each_period():
    dataset = {
        "basis": torch.ones((1, 2, 1)),
        "period_basis": torch.tensor([[[[1.0], [1.0]], [[3.0], [3.0]]]]),
        "target": torch.zeros((1, 2)), "theta": torch.zeros((1, 2)),
        "theta_dot": torch.zeros((1, 2)), "theta_ddot": torch.zeros((1, 2)),
        "motion_scales": torch.ones(3), "torque_scale": 1.0,
    }
    parameters = {
        "w1": torch.zeros((12, 1)), "b1": torch.zeros(1),
        "w2": torch.zeros((1, 1)), "b2": torch.zeros(1),
    }
    torque, _ = rollout(parameters, dataset, torch.tensor([2.0]), 0.0, 2)
    np.testing.assert_allclose(torque[0, 0].numpy(), [2.0, 2.0])
    np.testing.assert_allclose(torque[0, 1].numpy(), 3.0 * np.log(2.0) * np.ones(2))


def test_rollout_clips_predictions_to_per_spring_physical_bounds():
    dataset = {
        "basis": torch.ones((1, 2, 2)), "target": torch.zeros((1, 2)),
        "theta": torch.zeros((1, 2)), "theta_dot": torch.zeros((1, 2)),
        "theta_ddot": torch.zeros((1, 2)), "motion_scales": torch.ones(3),
        "torque_scale": 1.0,
    }
    parameters = {
        "w1": torch.zeros((12, 1)), "b1": torch.zeros(1),
        "w2": torch.zeros((1, 2)), "b2": torch.tensor([-100.0, 100.0]),
    }
    lower, upper = torch.tensor([5.0, 7.0]), torch.tensor([50.0, 70.0])
    _, stiffness = rollout(
        parameters, dataset, torch.tensor([10.0, 14.0]), 0.0, 2,
        stiffness_lower=lower, stiffness_upper=upper,
    )
    np.testing.assert_allclose(stiffness[0, 1].numpy(), [5.0, 70.0])
