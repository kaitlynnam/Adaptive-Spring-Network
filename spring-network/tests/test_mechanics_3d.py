from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from mechanics_3d import (  # noqa: E402
    load_spatial_topology,
    prescribed_positions,
    spring_state,
    torque_and_residual,
)


class SpatialMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            PROJECT_ROOT
            / "topologies"
            / "spatial"
            / "internal_fan_3d_30_spring_optimized.json"
        )

    def test_topology_is_genuinely_spatial_and_connected(self):
        topology = load_spatial_topology(self.path)
        self.assertEqual(topology["local_positions"].shape[1], 3)
        self.assertEqual(len(topology["spring_a"]), 30)
        self.assertEqual(len(topology["internal_indices"]), 6)
        self.assertGreater(
            float(torch.max(topology["local_positions"][:, 2]) - torch.min(topology["local_positions"][:, 2])),
            0.4,
        )
        used = set(topology["spring_a"].tolist()) | set(topology["spring_b"].tolist())
        fixed = {
            index
            for index, kind in enumerate(topology["node_types"])
            if kind == "fixed"
        }
        self.assertTrue(fixed.issubset(used))

    def test_relaxation_reduces_internal_force_residual(self):
        topology = load_spatial_topology(self.path)
        theta = torch.tensor([-0.5, 0.0, 0.5])
        stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(3, 1)
        _, shallow, _ = torque_and_residual(topology, theta, stiffness, 2)
        torque, deeper, _ = torque_and_residual(topology, theta, stiffness, 80)
        self.assertTrue(torch.isfinite(torque).all())
        self.assertLess(float(torch.mean(deeper)), float(torch.mean(shallow)))

    def test_joint_rotates_about_configured_y_axis(self):
        topology = load_spatial_topology(self.path)
        self.assertTrue(torch.allclose(
            topology["joint_axis"], torch.tensor([0.0, 1.0, 0.0])
        ))
        from mechanics_3d import prescribed_positions
        theta = torch.tensor([0.0, 0.4])
        positions = prescribed_positions(topology, theta)
        moving = topology["limb2_indices"]
        self.assertTrue(torch.allclose(
            positions[0, moving, 1], positions[1, moving, 1]
        ))
        self.assertFalse(torch.allclose(
            positions[0, moving, 2], positions[1, moving, 2]
        ))

    def test_cubic_hardening_increases_force_without_changing_zero_mode(self):
        topology = load_spatial_topology(self.path)
        theta = torch.tensor([0.4])
        positions = prescribed_positions(topology, theta)
        stiffness = topology["initial_stiffness"].unsqueeze(0)
        topology["cubic_ratio"] = 0.0
        linear_force, _, _ = spring_state(topology, positions, stiffness)
        topology["cubic_ratio"] = 0.5
        topology["cubic_reference_extension"] = 0.6
        cubic_force, _, _ = spring_state(topology, positions, stiffness)
        self.assertTrue(
            torch.linalg.norm(cubic_force, dim=2).ge(
                torch.linalg.norm(linear_force, dim=2) - 1e-7
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
