from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from mechanics_3d import (  # noqa: E402
    load_spatial_topology,
    prescribed_positions,
    spring_energy,
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

    def test_force_tolerance_can_stop_relaxation_early(self):
        topology = load_spatial_topology(self.path)
        theta = torch.tensor([0.0])
        stiffness = topology["initial_stiffness"].unsqueeze(0)
        _, residual, positions, iterations = torque_and_residual(
            topology,
            theta,
            stiffness,
            relaxation_steps=100,
            force_tolerance=1e6,
            return_iterations=True,
        )
        self.assertEqual(iterations, 10)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertTrue(torch.isfinite(positions).all())

    def test_previous_internal_state_can_warm_start_next_angle(self):
        topology = load_spatial_topology(self.path)
        stiffness = topology["initial_stiffness"].unsqueeze(0)
        _, _, positions = torque_and_residual(
            topology, torch.tensor([0.0]), stiffness, relaxation_steps=20
        )
        internal = positions[:, topology["internal_indices"], :]
        torque, residual, _ = torque_and_residual(
            topology,
            torch.tensor([0.05]),
            stiffness,
            relaxation_steps=20,
            initial_internal=internal,
        )
        self.assertTrue(torch.isfinite(torque).all())
        self.assertTrue(torch.isfinite(residual).all())

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

    def test_generalized_power_laws_are_restoring_and_conservative(self):
        for power in (3, 4, 5):
            topology = load_spatial_topology(self.path)
            topology["nonlinear_power"] = power
            topology["nonlinear_ratio"] = 0.3
            topology["nonlinear_reference_extension"] = 0.6
            positions = prescribed_positions(topology, torch.tensor([0.4]))
            positions = positions.detach().requires_grad_(True)
            stiffness = topology["initial_stiffness"].unsqueeze(0)
            force, _, _ = spring_state(topology, positions, stiffness)
            energy = spring_energy(topology, positions, stiffness)
            gradient = torch.autograd.grad(energy, positions)[0]
            first_a = int(topology["spring_a"][0])
            # Energy gradient includes every incident spring, so compare the
            # complete analytical spring-force assembly instead of one edge.
            assembled = torch.zeros_like(positions)
            for index, (a, b) in enumerate(
                zip(topology["spring_a"].tolist(), topology["spring_b"].tolist())
            ):
                assembled[:, a, :] += force[:, index, :]
                assembled[:, b, :] -= force[:, index, :]
            self.assertTrue(torch.allclose(-gradient, assembled, rtol=2e-4, atol=2e-4))
            self.assertTrue(torch.isfinite(force[:, first_a:first_a + 1]).all())


if __name__ == "__main__":
    unittest.main()
