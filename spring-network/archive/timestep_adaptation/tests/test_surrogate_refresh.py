from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from topology_loader import load_network  # noqa: E402
import train_adaptive_dataset as trainer  # noqa: E402


@unittest.skipIf(trainer.torch is None, "PyTorch is not installed")
class SurrogateRefreshTests(unittest.TestCase):
    def test_relaxed_components_sum_to_reported_torque(self):
        topology_path = (
            PROJECT_ROOT
            / "topologies"
            / "adaptive_stiffness"
            / "internal_fan_20_spring_model.json"
        )
        network, _ = load_network(topology_path)
        torch = trainer.torch
        topology = trainer.torch_topology_data(network, torch.device("cpu"))
        theta = torch.tensor([-0.6, 0.0, 0.6], dtype=torch.float32)
        stiffness = torch.tensor(
            np.vstack(
                [
                    np.linspace(40.0, 500.0, len(network.springs)),
                    np.linspace(500.0, 40.0, len(network.springs)),
                    np.full(len(network.springs), 220.0),
                ]
            ),
            dtype=torch.float32,
        )

        components = trainer.torch_torque_components_batch(
            topology, theta, stiffness, True, 8, 0.03
        )
        torque = trainer.torch_torque_batch(
            topology, theta, stiffness, True, 8, 0.03
        )

        np.testing.assert_allclose(
            components.detach().numpy().sum(axis=1),
            torque.detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
