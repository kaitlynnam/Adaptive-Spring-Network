from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import forward, initialize_model  # noqa: E402
from train_profile_conditioned_passive import (  # noqa: E402
    distill_profile_model,
    expand_profile_stiffness,
    optimize_profile_stiffness_oracle,
    profile_energy_burden,
    summary_rows,
    surrogate_torque,
)
from benchmark_profile_passive_3d import (  # noqa: E402
    relaxed_spatial_profile_torque,
)
from mechanics_3d import load_spatial_topology  # noqa: E402
from train_profile_conditioned_passive_3d import (  # noqa: E402
    correction_indices,
    refresh_mlp_spatial_basis,
    subset_profile_dataset,
)


class ProfileConditionedPassiveTests(unittest.TestCase):
    def test_one_profile_stiffness_is_constant_across_all_samples(self):
        stiffness = np.asarray([[10.0, 20.0], [30.0, 40.0]])
        schedule = expand_profile_stiffness(stiffness, samples=7)
        self.assertEqual(schedule.shape, (2, 7, 2))
        np.testing.assert_array_equal(schedule[:, 0, :], stiffness)
        np.testing.assert_array_equal(np.diff(schedule, axis=1), 0.0)

    def test_identical_profiles_produce_identical_stiffness(self):
        rng = np.random.default_rng(3)
        model = initialize_model(rng, 10, 8, 3, np.full(3, 50.0), 1.0, 100.0)
        descriptor = rng.normal(size=10)
        features = np.vstack((descriptor, descriptor))
        stiffness, _ = forward(model, features, 1.0, 100.0)
        np.testing.assert_array_equal(stiffness[0], stiffness[1])

    def test_profile_stiffness_is_used_for_every_torque_sample(self):
        dataset = {
            "samples_per_profile": 3,
            "basis": np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        }
        torque = surrogate_torque(np.asarray([[10.0, 20.0]]), dataset)
        np.testing.assert_array_equal(torque, [[50.0, 110.0, 170.0]])

    def test_invalid_stiffness_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            expand_profile_stiffness(np.ones(4), samples=3)

    def test_perfect_spring_assistance_has_full_offload(self):
        profiles = [{"name": "p", "family": "test"}]
        dataset = {
            "target": np.asarray([[2.0, 2.0, 2.0]]),
            "theta_dot": np.asarray([[1.0, 1.0, 1.0]]),
            "t": np.asarray([[0.0, 0.5, 1.0]]),
        }
        rows = summary_rows(
            profiles, dataset, dataset["target"], np.asarray([[10.0]])
        )
        self.assertAlmostEqual(rows[0]["offload_pct"], 100.0)
        self.assertAlmostEqual(rows[0]["assisted_energy_burden_j"], 0.0)

    def test_energy_burden_integrates_over_time(self):
        burden = profile_energy_burden(
            np.asarray([0.0, 1.0]),
            np.asarray([2.0, 2.0]),
            np.asarray([1.0, 1.0]),
            motoring_efficiency=1.0,
            regen_efficiency=0.0,
        )
        self.assertAlmostEqual(burden, 2.0)

    def test_oracle_improves_simple_surrogate_fit(self):
        dataset = {
            "basis": np.asarray([[[1.0], [2.0], [3.0]]]),
            "target": np.asarray([[5.0, 10.0, 15.0]]),
            "theta_dot": np.ones((1, 3)),
        }
        stiffness = optimize_profile_stiffness_oracle(
            dataset,
            initial_k=np.asarray([1.0]),
            iterations=200,
            learning_rate=0.1,
            min_k=0.1,
            max_k=10.0,
            energy_weight=0.0,
            motoring_efficiency=1.0,
            regen_efficiency=0.0,
            device="cpu",
            progress_interval=0,
        )
        self.assertAlmostEqual(stiffness[0, 0], 5.0, delta=0.1)

    def test_distillation_learns_profile_specific_labels(self):
        features = np.vstack((np.full(10, -1.0), np.full(10, 1.0)))
        dataset = {
            "profile_features": features,
            "basis": np.zeros((2, 1, 1)),
        }
        labels = np.asarray([[2.0], [8.0]])
        model = distill_profile_model(
            dataset,
            labels,
            initial_k=np.asarray([5.0]),
            hidden_dim=8,
            iterations=300,
            learning_rate=0.03,
            min_k=1.0,
            max_k=10.0,
            seed=7,
            device="cpu",
            progress_interval=0,
        )
        predicted, _ = forward(model, features, 1.0, 10.0)
        np.testing.assert_allclose(predicted, labels, atol=0.15)

    def test_unbounded_oracle_can_exceed_nominal_maximum(self):
        dataset = {
            "basis": np.asarray([[[1.0], [2.0]]]),
            "target": np.asarray([[20.0, 40.0]]),
            "theta_dot": np.ones((1, 2)),
        }
        stiffness = optimize_profile_stiffness_oracle(
            dataset,
            initial_k=np.asarray([2.0]),
            iterations=500,
            learning_rate=0.2,
            min_k=0.0,
            max_k=5.0,
            energy_weight=0.0,
            motoring_efficiency=1.0,
            regen_efficiency=0.0,
            device="cpu",
            progress_interval=0,
            unbounded_stiffness=True,
        )
        self.assertGreater(stiffness[0, 0], 5.0)
        self.assertAlmostEqual(stiffness[0, 0], 20.0, delta=0.2)

    def test_spatial_rollout_holds_profile_stiffness_constant(self):
        path = (
            PROJECT_ROOT
            / "topologies"
            / "spatial"
            / "internal_fan_3d_24_spring_sparse.json"
        )
        topology = load_spatial_topology(path, "cpu")
        stiffness = topology["initial_stiffness"].detach().numpy()[None, :]
        dataset = {
            "theta": np.asarray([[-0.2, 0.0, 0.2]]),
            "target": np.zeros((1, 3)),
            "samples_per_profile": 3,
        }
        torque, residual = relaxed_spatial_profile_torque(
            dataset, topology, stiffness, relaxation_steps=2, batch_size=8
        )
        self.assertEqual(torque.shape, (1, 3))
        self.assertEqual(residual.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(torque)))

    def test_profile_subset_preserves_complete_profile_rows(self):
        dataset = {
            "profile_features": np.arange(30).reshape(3, 10),
            "theta": np.arange(12).reshape(3, 4),
            "target": np.arange(12).reshape(3, 4),
            "samples_per_profile": 4,
        }
        subset = subset_profile_dataset(dataset, [2, 0])
        np.testing.assert_array_equal(subset["theta"], dataset["theta"][[2, 0]])
        self.assertEqual(subset["samples_per_profile"], 4)

    def test_default_correction_indices_use_every_profile_and_sample(self):
        profiles, samples = correction_indices(
            6000, 160, 0, 0, np.random.default_rng(1)
        )
        np.testing.assert_array_equal(profiles, np.arange(6000))
        np.testing.assert_array_equal(samples, np.arange(160))

    def test_positive_correction_limits_are_explicit_subsets(self):
        profiles, samples = correction_indices(
            6000, 160, 512, 64, np.random.default_rng(1)
        )
        self.assertEqual(len(profiles), 512)
        self.assertEqual(len(samples), 64)
        self.assertEqual(len(np.unique(profiles)), 512)


if __name__ == "__main__":
    unittest.main()
