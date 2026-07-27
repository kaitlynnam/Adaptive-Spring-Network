from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from topology_loader import load_network  # noqa: E402
from train_preload_network import (  # noqa: E402
    full_relaxed_preload_torque,
    preload_adjustment_energy,
)


class PreloadNetEnergyTests(unittest.TestCase):
    def test_adjustment_work_uses_current_extension_and_charges_only_positive_work(self):
        schedule = torch.tensor([[[0.1], [0.2], [0.1]]])
        lengths = torch.tensor([[[1.0], [1.0], [1.0]]])
        rest = torch.tensor([[[1.0]]])
        stiffness = torch.tensor([10.0])
        cubic = torch.zeros(1)

        work = preload_adjustment_energy(
            schedule, lengths, rest, stiffness, cubic
        )

        # At the second sample, stretch rises from 0.1 to 0.2:
        # 0.5 * 10 * (0.2^2 - 0.1^2) = 0.15 J.
        self.assertAlmostEqual(float(work[0, 0, 0]), 0.15, places=6)
        # Returning from 0.2 to 0.1 releases energy, which is not recovered.
        self.assertEqual(float(work[0, 1, 0]), 0.0)

    def test_local_preload_sensitivity_matches_small_torque_perturbation(self):
        topology_path = (
            PROJECT_ROOT / "topologies" / "preload" / "preload_fan_soft_015_long150.json"
        )
        network, _ = load_network(topology_path)
        preload = torch.full((1, len(network.springs)), 0.4)
        dataset = {"theta": torch.tensor([0.2]).numpy()}
        torque, sensitivity = full_relaxed_preload_torque(
            dataset,
            preload,
            topology_path,
            torch.device("cpu"),
            batch_size=1,
            relaxation_steps=0,
            return_sensitivity=True,
        )
        delta = 1e-5
        perturbed = preload.clone()
        perturbed[:, 0] += delta
        perturbed_torque = full_relaxed_preload_torque(
            dataset,
            perturbed,
            topology_path,
            torch.device("cpu"),
            batch_size=1,
            relaxation_steps=0,
        )
        finite_difference = (perturbed_torque - torque) / delta
        self.assertAlmostEqual(
            float(sensitivity[0, 0]), float(finite_difference[0]), delta=0.05
        )


if __name__ == "__main__":
    unittest.main()
