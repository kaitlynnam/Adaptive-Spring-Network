from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from geometry import Node, update_node_position
from physics import Spring, spring_force, torque_about_origin
from visualize import plot_network


@dataclass
class SpringNetwork:
    nodes: dict[str, Node]
    springs: list[Spring]

    def update_positions(self, theta, relax_internal=False, relaxation_solver="energy"):
        """Compute the current world position of every node for a joint angle."""
        for node in self.nodes.values():
            if relax_internal and node.type == "internal" and node.current_position is not None:
                continue
            update_node_position(node, theta)

        if relax_internal:
            self.solve_internal_equilibrium(solver=relaxation_solver)

    def initialize_rest_lengths(self, theta=0.0):
        """Use the network geometry at theta as the unstretched spring state."""
        self.update_positions(theta)
        for spring in self.springs:
            a = self.nodes[spring.node_a].current_position
            b = self.nodes[spring.node_b].current_position
            spring.rest_length = np.linalg.norm(b - a)

    def compute_forces(self):
        """Compute spring forces and accumulate equal/opposite node forces."""
        forces = {name: np.zeros(2, dtype=float) for name in self.nodes}
        spring_results = []

        for spring in self.springs:
            if spring.rest_length is None:
                raise ValueError(f"Spring {spring.node_a}-{spring.node_b} has no rest length.")

            node_a = self.nodes[spring.node_a]
            node_b = self.nodes[spring.node_b]
            force_on_a, current_length, stretch = spring_force(
                node_a.current_position,
                node_b.current_position,
                spring.stiffness_k,
                spring.rest_length,
            )

            forces[node_a.name] += force_on_a
            forces[node_b.name] -= force_on_a
            spring_results.append(
                {
                    "spring": spring,
                    "current_length": current_length,
                    "stretch": stretch,
                    "force_on_a": force_on_a,
                }
            )

        return forces, spring_results

    def internal_nodes(self):
        return [node for node in self.nodes.values() if node.type == "internal"]

    def solve_internal_equilibrium(self, solver="energy"):
        if solver == "energy":
            result = self.minimize_internal_energy()
            if result["success"]:
                return result
            return self.relax_internal_nodes()
        if solver == "force":
            return self.relax_internal_nodes()
        raise ValueError(f"Unknown internal relaxation solver {solver!r}. Use 'energy' or 'force'.")

    def minimize_internal_energy(self, max_iterations=80, tolerance=1e-7):
        """Place internal nodes by minimizing total spring potential energy.

        Fixed, limb1, and limb2 nodes are prescribed by the joint geometry. The
        optimizer changes only internal node coordinates. At equilibrium, the
        energy gradient is zero, which is equivalent to near-zero net spring
        force on every internal node.
        """
        internal_nodes = self.internal_nodes()
        if not internal_nodes:
            return {"success": True, "max_force": 0.0, "iterations": 0}

        node_order = [node.name for node in internal_nodes]
        x0 = np.concatenate([self.nodes[name].current_position for name in node_order])

        def set_internal_positions(x):
            for index, name in enumerate(node_order):
                self.nodes[name].current_position = x[2 * index : 2 * index + 2].copy()

        def energy_and_gradient(x):
            set_internal_positions(x)
            energy = 0.0
            internal_forces = {name: np.zeros(2, dtype=float) for name in node_order}

            for spring in self.springs:
                if spring.rest_length is None:
                    raise ValueError(f"Spring {spring.node_a}-{spring.node_b} has no rest length.")

                node_a = self.nodes[spring.node_a]
                node_b = self.nodes[spring.node_b]
                force_on_a, current_length, stretch = spring_force(
                    node_a.current_position,
                    node_b.current_position,
                    spring.stiffness_k,
                    spring.rest_length,
                )
                energy += 0.5 * spring.stiffness_k * stretch**2

                if node_a.name in internal_forces:
                    internal_forces[node_a.name] += force_on_a
                if node_b.name in internal_forces:
                    internal_forces[node_b.name] -= force_on_a

            gradient = np.concatenate([-internal_forces[name] for name in node_order])
            return energy, gradient

        result = minimize(
            energy_and_gradient,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iterations,
                "ftol": tolerance,
                "gtol": tolerance,
                "maxls": 30,
            },
        )
        set_internal_positions(result.x)
        forces, _ = self.compute_forces()
        max_force = max(float(np.linalg.norm(forces[name])) for name in node_order)
        return {
            "success": bool(result.success or max_force < 1e-4),
            "max_force": max_force,
            "iterations": int(result.nit),
        }

    def relax_internal_nodes(self, max_iterations=120, tolerance=1e-4, step_size=0.015, max_step=0.03):
        """Move internal nodes toward quasi-static force equilibrium.

        Fixed, limb1, and limb2 nodes are prescribed by the joint geometry. Only
        nodes marked "internal" are moved. The spring force is the negative
        energy gradient, so stepping along net force reduces spring energy for
        sufficiently small steps.
        """
        internal_nodes = self.internal_nodes()
        if not internal_nodes:
            return {"success": True, "max_force": 0.0, "iterations": 0}

        max_force = float("inf")
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            forces, _ = self.compute_forces()
            max_force = max(float(np.linalg.norm(forces[node.name])) for node in internal_nodes)
            if max_force < tolerance:
                break

            for node in internal_nodes:
                step = step_size * forces[node.name]
                step_norm = np.linalg.norm(step)
                if step_norm > max_step:
                    step = step / step_norm * max_step
                node.current_position = node.current_position + step

        return {"success": max_force < tolerance, "max_force": max_force, "iterations": iterations}

    def compute_torque(self, forces, node_types=("limb2",)):
        """Compute net torque about the joint from selected node forces.

        For this first joint model, the most useful value is the torque applied
        to limb 2 by the springs. Future versions could separately report
        anchor reactions, internal node relaxation, or actuator torques.
        """
        torque = 0.0
        for name, force in forces.items():
            node = self.nodes[name]
            if node.type in node_types:
                torque += torque_about_origin(node.current_position, force)
        return torque

    def evaluate(self, theta, relax_internal=False, relaxation_solver="energy"):
        self.update_positions(theta, relax_internal=relax_internal, relaxation_solver=relaxation_solver)
        forces, spring_results = self.compute_forces()
        torque = self.compute_torque(forces)
        return forces, spring_results, torque

    def plot(self, theta, forces=None, show_forces=True, ax=None, relax_internal=False):
        return plot_network(
            self,
            theta,
            forces=forces,
            show_forces=show_forces,
            ax=ax,
            relax_internal=relax_internal,
        )
