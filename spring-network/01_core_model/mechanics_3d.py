"""Batched spatial spring mechanics for a single configurable-axis revolute joint."""

from pathlib import Path
import json

import numpy as np
import torch


def load_spatial_topology(path, device="cpu"):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    names = [node["name"] for node in data["nodes"]]
    index = {name: i for i, name in enumerate(names)}
    node_types = [node["type"] for node in data["nodes"]]
    positions = np.asarray([node["position"] for node in data["nodes"]], dtype=float)
    if positions.shape[1] != 3:
        raise ValueError("Spatial topology nodes must have [x, y, z] positions")
    a = np.asarray([index[s["node_a"]] for s in data["springs"]], dtype=np.int64)
    b = np.asarray([index[s["node_b"]] for s in data["springs"]], dtype=np.int64)
    stiffness = np.asarray([s["stiffness_k"] for s in data["springs"]], dtype=float)
    geometric_length = np.linalg.norm(positions[b] - positions[a], axis=1)
    scale = float(data.get("rest_length_scale", 1.0))
    rest = np.asarray(
        [s.get("rest_length", length * scale) for s, length in zip(data["springs"], geometric_length)],
        dtype=float,
    )
    tensor_device = torch.device(device)
    joint_axis = torch.as_tensor(
        data.get("joint_axis", [0.0, 0.0, 1.0]),
        dtype=torch.float32,
        device=tensor_device,
    )
    axis_norm = torch.linalg.norm(joint_axis)
    if float(axis_norm) < 1e-9:
        raise ValueError("joint_axis must be a nonzero 3D vector")
    joint_axis = joint_axis / axis_norm
    return {
        "data": data,
        "names": names,
        "node_types": node_types,
        "local_positions": torch.as_tensor(positions, dtype=torch.float32, device=tensor_device),
        "spring_a": torch.as_tensor(a, dtype=torch.long, device=tensor_device),
        "spring_b": torch.as_tensor(b, dtype=torch.long, device=tensor_device),
        "rest_lengths": torch.as_tensor(rest, dtype=torch.float32, device=tensor_device),
        "initial_stiffness": torch.as_tensor(stiffness, dtype=torch.float32, device=tensor_device),
        "joint_axis": joint_axis,
        "bearing_radius": float(data.get("bearing_radius", 0.0)),
        "bearing_clearance": float(data.get("bearing_clearance", 0.0)),
        "bearing_half_length": float(data.get("bearing_half_length", 0.0)),
        "bearing_collision_penalty": float(
            data.get("bearing_collision_penalty", 0.0)
        ),
        "nonlinear_power": int(data.get("nonlinear_power", 1)),
        "nonlinear_ratio": float(data.get("nonlinear_ratio", 0.0)),
        "nonlinear_reference_extension": float(
            data.get("nonlinear_reference_extension", 0.6)
        ),
        "internal_indices": torch.as_tensor(
            [i for i, kind in enumerate(node_types) if kind == "internal"],
            dtype=torch.long, device=tensor_device,
        ),
        "limb2_indices": torch.as_tensor(
            [
                i for i, kind in enumerate(node_types)
                if kind in ("limb2", "skin2")
            ],
            dtype=torch.long, device=tensor_device,
        ),
    }


def prescribed_positions(topology, theta):
    local = topology["local_positions"]
    positions = local.unsqueeze(0).repeat(len(theta), 1, 1)
    c, s = torch.cos(theta), torch.sin(theta)
    axis = topology["joint_axis"]
    for i, kind in enumerate(topology["node_types"]):
        if kind in ("limb2", "skin2"):
            point = local[i]
            cross = torch.linalg.cross(axis, point)
            projection = torch.dot(axis, point)
            positions[:, i, :] = (
                c[:, None] * point
                + s[:, None] * cross
                + (1.0 - c)[:, None] * projection * axis
            )
    return positions


def spring_state(topology, positions, stiffness):
    a, b = topology["spring_a"], topology["spring_b"]
    delta = positions[:, b, :] - positions[:, a, :]
    length = torch.linalg.norm(delta, dim=2).clamp_min(1e-9)
    direction = delta / length.unsqueeze(2)
    stretch = length - topology["rest_lengths"].unsqueeze(0)
    force_scale = stiffness * stretch
    nonlinear_ratio = topology.get("nonlinear_ratio", 0.0)
    nonlinear_power = topology.get("nonlinear_power", 1)
    nonlinear_reference = max(
        topology.get("nonlinear_reference_extension", 0.6), 1e-9
    )
    if nonlinear_ratio:
        force_scale = force_scale * (
            1.0
            + nonlinear_ratio
            * (torch.abs(stretch) / nonlinear_reference) ** (nonlinear_power - 1)
        )
    force_on_a = force_scale.unsqueeze(2) * direction
    return force_on_a, length, stretch


def spring_energy(topology, positions, stiffness):
    _, _, stretch = spring_state(topology, positions, stiffness)
    energy = torch.sum(0.5 * stiffness * stretch**2)
    nonlinear_ratio = topology.get("nonlinear_ratio", 0.0)
    nonlinear_power = topology.get("nonlinear_power", 1)
    nonlinear_reference = max(
        topology.get("nonlinear_reference_extension", 0.6), 1e-9
    )
    if nonlinear_ratio:
        energy = energy + torch.sum(
            stiffness
            * nonlinear_ratio
            * torch.abs(stretch) ** (nonlinear_power + 1)
            / (
                (nonlinear_power + 1)
                * nonlinear_reference ** (nonlinear_power - 1)
            )
        )
    radius = topology["bearing_radius"] + topology["bearing_clearance"]
    penalty = topology["bearing_collision_penalty"]
    if radius > 0.0 and penalty > 0.0:
        a, b = topology["spring_a"], topology["spring_b"]
        start, stop = positions[:, a, :], positions[:, b, :]
        fractions = torch.linspace(
            0.05, 0.95, 19, dtype=positions.dtype, device=positions.device
        )
        points = (
            start[:, :, None, :]
            + fractions[None, None, :, None]
            * (stop - start)[:, :, None, :]
        )
        axis = topology["joint_axis"]
        axial = torch.sum(points * axis, dim=3)
        radial_vector = points - axial[:, :, :, None] * axis
        radial = torch.linalg.norm(radial_vector, dim=3)
        inside_length = (
            torch.abs(axial) < topology["bearing_half_length"]
        ).to(positions.dtype)
        penetration = torch.relu(radius - radial)
        energy = energy + penalty * torch.sum(inside_length * penetration**2)
    return energy


def relax_positions(
    topology,
    prescribed,
    stiffness,
    steps=160,
    learning_rate=0.015,
    initial_internal=None,
    force_tolerance=None,
    min_steps=10,
    return_iterations=False,
):
    internal = topology["internal_indices"]
    if internal.numel() == 0 or steps == 0:
        return (prescribed, 0) if return_iterations else prescribed
    if initial_internal is None:
        initial_internal = prescribed[:, internal, :]
    if initial_internal.shape != prescribed[:, internal, :].shape:
        raise ValueError("initial_internal must have shape [batch, internal_nodes, 3]")
    values = initial_internal.detach().clone().requires_grad_(True)
    use_lbfgs_polish = steps >= 600
    adam_steps = min(300, steps) if use_lbfgs_polish else steps
    optimizer = torch.optim.Adam([values], lr=learning_rate)
    completed_steps = 0
    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        positions = prescribed.clone()
        positions[:, internal, :] = values
        spring_energy(topology, positions, stiffness).backward()
        optimizer.step()
        completed_steps += 1
        if (
            force_tolerance is not None
            and completed_steps >= min_steps
            and (completed_steps % 5 == 0 or completed_steps == adam_steps)
        ):
            probe = prescribed.clone()
            probe[:, internal, :] = values
            gradient = torch.autograd.grad(
                spring_energy(topology, probe, stiffness), values
            )[0]
            residual = torch.max(torch.linalg.norm(gradient, dim=2))
            if float(residual.detach()) <= force_tolerance:
                break
    if use_lbfgs_polish:
        # Restarting discards the early-stage moments while preserving the
        # configured learning rate throughout mechanics relaxation.
        optimizer = torch.optim.Adam([values], lr=learning_rate)
        polish_steps = steps - adam_steps
        for _ in range(polish_steps):
            optimizer.zero_grad(set_to_none=True)
            positions = prescribed.clone()
            positions[:, internal, :] = values
            loss = spring_energy(topology, positions, stiffness)
            loss.backward()
            optimizer.step()
            completed_steps += 1
            if (
                force_tolerance is not None
                and completed_steps >= min_steps
                and completed_steps % 5 == 0
            ):
                probe = prescribed.clone()
                probe[:, internal, :] = values
                gradient = torch.autograd.grad(
                    spring_energy(topology, probe, stiffness), values
                )[0]
                residual = torch.max(torch.linalg.norm(gradient, dim=2))
                if float(residual.detach()) <= force_tolerance:
                    break
    positions = prescribed.clone()
    positions[:, internal, :] = values.detach()
    return (positions, completed_steps) if return_iterations else positions


def torque_components(topology, theta, stiffness, relaxation_steps=160):
    components, _, _ = torque_components_and_residual(
        topology, theta, stiffness, relaxation_steps
    )
    return components


def torque_components_and_residual(
    topology,
    theta,
    stiffness,
    relaxation_steps=160,
    initial_internal=None,
    force_tolerance=None,
    return_iterations=False,
):
    """Return per-spring joint torque and the relaxed internal-force residual."""
    relaxed = relax_positions(
        topology,
        prescribed_positions(topology, theta),
        stiffness,
        relaxation_steps,
        initial_internal=initial_internal,
        force_tolerance=force_tolerance,
        return_iterations=return_iterations,
    )
    if return_iterations:
        positions, completed_steps = relaxed
    else:
        positions, completed_steps = relaxed, None
    force_on_a, _, _ = spring_state(topology, positions, stiffness)
    internal = topology["internal_indices"]
    if internal.numel():
        with torch.enable_grad():
            probe = positions.detach().clone().requires_grad_(True)
            total_energy = spring_energy(topology, probe, stiffness.detach())
            total_force = -torch.autograd.grad(total_energy, probe)[0]
        residual = torch.max(
            torch.linalg.norm(total_force[:, internal, :], dim=2), dim=1
        ).values
    else:
        residual = torch.zeros(len(theta), device=theta.device)
    result = (
        torque_components_from_state(topology, positions, force_on_a),
        residual,
        positions,
    )
    return (*result, completed_steps) if return_iterations else result


def torque_and_residual(
    topology,
    theta,
    stiffness,
    relaxation_steps=160,
    initial_internal=None,
    force_tolerance=None,
    return_iterations=False,
):
    result = torque_components_and_residual(
        topology,
        theta,
        stiffness,
        relaxation_steps,
        initial_internal=initial_internal,
        force_tolerance=force_tolerance,
        return_iterations=return_iterations,
    )
    if return_iterations:
        components, residual, positions, iterations = result
        return torch.sum(components, dim=1), residual, positions, iterations
    components, residual, positions = result
    return torch.sum(components, dim=1), residual, positions


def torque_components_from_state(topology, positions, force_on_a):
    components = torch.zeros(
        (len(positions), len(topology["spring_a"])),
        dtype=positions.dtype,
        device=positions.device,
    )
    limb2 = set(int(i) for i in topology["limb2_indices"].detach().cpu().numpy())
    axis = topology["joint_axis"]
    for spring_index, (a, b) in enumerate(
        zip(topology["spring_a"].tolist(), topology["spring_b"].tolist())
    ):
        if a in limb2:
            moment = torch.cross(
                positions[:, a, :], force_on_a[:, spring_index, :], dim=1
            )
            components[:, spring_index] += torch.sum(moment * axis, dim=1)
        if b in limb2:
            moment = torch.cross(
                positions[:, b, :], -force_on_a[:, spring_index, :], dim=1
            )
            components[:, spring_index] += torch.sum(moment * axis, dim=1)
    return components
