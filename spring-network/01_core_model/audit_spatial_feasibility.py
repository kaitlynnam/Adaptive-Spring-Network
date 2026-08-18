"""Audit geometric feasibility of the spatial joint over its operating range."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from mechanics_3d import load_spatial_topology, prescribed_positions, relax_positions


DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"
)


def segment_distance(p0, p1, q0, q1):
    """Minimum distance between two finite 3D line segments."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d, e = np.dot(u, w), np.dot(v, w)
    denominator = a * c - b * b
    if denominator < 1e-12:
        s = 0.0
        t = np.clip(e / max(c, 1e-12), 0.0, 1.0)
    else:
        s = np.clip((b * e - c * d) / denominator, 0.0, 1.0)
        t = np.clip((a * e - b * d) / denominator, 0.0, 1.0)
    # Recompute once after clamping so endpoint-to-segment cases are correct.
    t = np.clip((b * s + e) / max(c, 1e-12), 0.0, 1.0)
    s = np.clip((b * t - d) / max(a, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(w + s * u - t * v))


def audit(topology_path, relaxation_steps=300, angle_samples=19, device="cpu",
          rest_length_scale=None):
    topology = load_spatial_topology(topology_path, device)
    configured_rest_length_scale = float(
        topology["data"].get("rest_length_scale", 1.0)
    )
    active_rest_length_scale = (
        configured_rest_length_scale
        if rest_length_scale is None else float(rest_length_scale)
    )
    if active_rest_length_scale <= 0.0:
        raise ValueError("rest_length_scale must be positive")
    topology["rest_lengths"] *= (
        active_rest_length_scale / configured_rest_length_scale
    )
    theta = torch.linspace(
        -np.pi / 4, np.pi / 4, angle_samples,
        dtype=torch.float32, device=topology["local_positions"].device,
    )
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(angle_samples, 1)
    positions = relax_positions(
        topology,
        prescribed_positions(topology, theta),
        stiffness,
        steps=relaxation_steps,
    ).detach().cpu().numpy()
    spring_a = topology["spring_a"].cpu().numpy()
    spring_b = topology["spring_b"].cpu().numpy()
    axis = topology["joint_axis"].detach().cpu().numpy()
    radius = topology["bearing_radius"]
    bearing_required_centerline_radius = (
        radius + float(topology["data"].get("bearing_clearance", 0.0))
        + float(topology["data"].get("spring_radius", 0.015))
    )
    half_length = topology["bearing_half_length"]
    fractions = np.linspace(0.001, 0.999, 1001)
    minimum_bearing_radius = np.inf
    bearing_intersections = 0
    bearing_intersection_examples = []
    bearing_collision_counts = np.zeros(len(topology["spring_a"]), dtype=int)
    bearing_collision_angles = []
    limb_intersections = 0
    limb_intersection_examples = []
    minimum_spring_length = np.inf
    skin_containment_violations = 0
    skin_containment_examples = []
    skin_radius = topology["data"].get("skin_radius")
    joint_boot_radius = topology["data"].get("joint_boot_radius", skin_radius)
    limb_thickness_scale = float(topology["data"].get("limb_thickness_scale", 1.0))
    spring_clearance = float(topology["data"].get("spring_clearance", 0.032))
    spring_clearance_violations = 0
    spring_clearance_examples = []
    minimum_spring_to_spring_clearance = np.inf
    spring_collision_counts = np.zeros(len(spring_a), dtype=int)
    spring_pair_collision_counts = {}
    spring_collision_angles = []
    support_radius = float(topology["data"].get("support_radius", 0.012))
    spring_radius = float(topology["data"].get("spring_radius", 0.015))
    spring_support_violations = 0
    spring_support_examples = []
    minimum_spring_support_clearance = np.inf
    local_positions = topology["local_positions"].detach().cpu().numpy()

    def inside_limb(points, limb_type, angle):
        local = points.copy()
        if limb_type == "limb2":
            c, s = np.cos(angle), np.sin(angle)
            x = c * points[:, 0] - s * points[:, 2]
            z = s * points[:, 0] + c * points[:, 2]
            local[:, 0], local[:, 2] = x, z
            start_x, stop_x, outer = 0.065, 1.11, 1.11
        else:
            start_x, stop_x, outer = -0.98, -0.065, 0.98
        x = local[:, 0]
        along_link = (x >= start_x) & (x <= stop_x)
        scale = np.abs(x) / outer
        half_y = limb_thickness_scale * (0.035 + 0.030 * scale)
        half_z = limb_thickness_scale * (0.055 + 0.045 * scale)
        return (
            along_link
            & (np.abs(local[:, 1]) < half_y - 0.002)
            & (np.abs(local[:, 2]) < half_z - 0.002)
        )

    for state_index, state in enumerate(positions):
        angle = float(theta[state_index].detach().cpu())
        # Ignore the first/last 4% at mounting eyelets, where separate attachment
        # hardware can fan out springs that intentionally share a node.
        trimmed = []
        for a, b in zip(spring_a, spring_b):
            delta = state[b] - state[a]
            trimmed.append((state[a] + 0.04 * delta, state[b] - 0.04 * delta))
        supports = []
        for node_index, kind in enumerate(topology["node_types"]):
            if kind not in ("limb1", "limb2"):
                continue
            if topology["data"].get("joint_nodes_on_limb_surface", False):
                continue
            local_node = local_positions[node_index]
            outer = 0.98 if kind == "limb1" else 1.11
            scale = abs(local_node[0]) / outer
            half_y = limb_thickness_scale * (0.035 + 0.030 * scale)
            half_z = limb_thickness_scale * (0.055 + 0.045 * scale)
            ellipse = np.sqrt(
                (local_node[1] / half_y) ** 2 + (local_node[2] / half_z) ** 2
            )
            base = local_node.copy()
            if ellipse > 1.0:
                base[1:] /= ellipse
            if kind == "limb2":
                c, s = np.cos(angle), np.sin(angle)
                bx, bz = base[0], base[2]
                base[0], base[2] = c * bx + s * bz, -s * bx + c * bz
            supports.append((node_index, base, state[node_index]))
        required_support_clearance = spring_radius + support_radius
        for spring_index, (start, stop) in enumerate(trimmed):
            endpoints = (int(spring_a[spring_index]), int(spring_b[spring_index]))
            for support_node, support_start, support_stop in supports:
                if support_node in endpoints:
                    continue
                distance = segment_distance(start, stop, support_start, support_stop)
                minimum_spring_support_clearance = min(
                    minimum_spring_support_clearance, distance
                )
                if distance < required_support_clearance:
                    spring_support_violations += 1
                    if len(spring_support_examples) < 16:
                        spring_support_examples.append({
                            "angle_degrees": float(np.degrees(angle)),
                            "spring_index": spring_index,
                            "support_node": topology["names"][support_node],
                            "clearance_m": distance,
                        })
        for first in range(len(trimmed)):
            for second in range(first + 1, len(trimmed)):
                if (
                    spring_a[first] in (spring_a[second], spring_b[second])
                    or spring_b[first] in (spring_a[second], spring_b[second])
                ):
                    # These springs intentionally meet at a common eyelet/node.
                    continue
                distance = segment_distance(
                    trimmed[first][0], trimmed[first][1],
                    trimmed[second][0], trimmed[second][1],
                )
                minimum_spring_to_spring_clearance = min(
                    minimum_spring_to_spring_clearance, distance
                )
                if distance < spring_clearance:
                    spring_clearance_violations += 1
                    spring_collision_counts[first] += 1
                    spring_collision_counts[second] += 1
                    pair = (first, second)
                    spring_pair_collision_counts[pair] = (
                        spring_pair_collision_counts.get(pair, 0) + 1
                    )
                    spring_collision_angles.append(float(np.degrees(angle)))
                    if len(spring_clearance_examples) < 16:
                        spring_clearance_examples.append({
                            "angle_degrees": float(np.degrees(angle)),
                            "spring_a": first,
                            "spring_b": second,
                            "clearance_m": distance,
                        })
        for spring_index, (a, b) in enumerate(zip(spring_a, spring_b)):
            delta = state[b] - state[a]
            minimum_spring_length = min(
                minimum_spring_length, float(np.linalg.norm(delta))
            )
            points = state[a] + fractions[:, None] * delta
            if skin_radius is not None:
                proximal_inside = (
                    (points[:, 0] >= -1.05) & (points[:, 0] <= 0.0)
                    & (np.hypot(points[:, 1], points[:, 2]) <= skin_radius + 1e-6)
                )
                c, s = np.cos(angle), np.sin(angle)
                distal_x = c * points[:, 0] - s * points[:, 2]
                distal_z = s * points[:, 0] + c * points[:, 2]
                distal_inside = (
                    (distal_x >= 0.0) & (distal_x <= 1.12)
                    & (np.hypot(points[:, 1], distal_z) <= skin_radius + 1e-6)
                )
                # A flexible joint boot closes the wedge between the two rigid
                # cylindrical skin halves as the distal half rotates.
                joint_boot_inside = (
                    np.linalg.norm(points, axis=1) <= joint_boot_radius + 1e-6
                )
                if not np.all(proximal_inside | distal_inside | joint_boot_inside):
                    skin_containment_violations += 1
                    if len(skin_containment_examples) < 12:
                        skin_containment_examples.append({
                            "angle_degrees": float(np.degrees(angle)),
                            "spring_index": spring_index,
                            "node_a": topology["names"][int(a)],
                            "node_b": topology["names"][int(b)],
                        })
            axial = points @ axis
            radial_vector = points - axial[:, None] * axis
            radial = np.linalg.norm(radial_vector, axis=1)
            within_bearing_length = np.abs(axial) < half_length
            if np.any(within_bearing_length):
                local_minimum = float(np.min(radial[within_bearing_length]))
                minimum_bearing_radius = min(minimum_bearing_radius, local_minimum)
                bearing_violation = (
                    local_minimum < bearing_required_centerline_radius
                )
                bearing_intersections += int(bearing_violation)
                if bearing_violation:
                    bearing_collision_counts[spring_index] += 1
                    bearing_collision_angles.append(float(np.degrees(angle)))
                if (
                    local_minimum < bearing_required_centerline_radius
                    and len(bearing_intersection_examples) < 20
                ):
                    bearing_intersection_examples.append({
                        "angle_degrees": float(np.degrees(angle)),
                        "spring_index": spring_index,
                        "node_a": topology["names"][int(a)],
                        "node_b": topology["names"][int(b)],
                        "minimum_radius_m": local_minimum,
                    })
            for limb_type in ("limb1", "limb2"):
                attached_a = topology["node_types"][int(a)] == limb_type
                attached_b = topology["node_types"][int(b)] == limb_type
                test_points = points
                if attached_a:
                    test_points = test_points[fractions >= 0.04]
                if attached_b:
                    test_points = test_points[fractions <= 0.96]
                if np.any(inside_limb(test_points, limb_type, angle)):
                    limb_intersections += 1
                    if len(limb_intersection_examples) < 12:
                        limb_intersection_examples.append({
                            "angle_degrees": float(np.degrees(angle)),
                            "spring_index": spring_index,
                            "limb": limb_type,
                            "node_a": topology["names"][int(a)],
                            "node_b": topology["names"][int(b)],
                        })

    used = set(spring_a.tolist()) | set(spring_b.tolist())
    all_nodes_used = len(used) == len(topology["names"])
    graph = {index: set() for index in range(len(topology["names"]))}
    spring_degree = {index: 0 for index in range(len(topology["names"]))}
    fixed_to_fixed_springs = 0
    same_rigid_body_springs = 0
    skin_to_skin_springs = 0
    rigid_body = {
        "limb1": "proximal",
        "skin1": "proximal",
        "limb2": "distal",
        "skin2": "distal",
    }
    for a, b in zip(spring_a, spring_b):
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
        spring_degree[int(a)] += 1
        spring_degree[int(b)] += 1
        if (
            topology["node_types"][int(a)] == "fixed"
            and topology["node_types"][int(b)] == "fixed"
        ):
            fixed_to_fixed_springs += 1
        body_a = rigid_body.get(topology["node_types"][int(a)])
        body_b = rigid_body.get(topology["node_types"][int(b)])
        if (
            topology["node_types"][int(a)] in ("skin1", "skin2")
            and topology["node_types"][int(b)] in ("skin1", "skin2")
        ):
            skin_to_skin_springs += 1
        if body_a is not None and body_a == body_b:
            same_rigid_body_springs += 1
    # Limb and skin nodes on each side share their rigid-body connection.
    for limb_types in (("limb1", "skin1"), ("limb2", "skin2")):
        rigid_nodes = [
            index
            for index, kind in enumerate(topology["node_types"])
            if kind in limb_types
        ]
        for first, second in zip(rigid_nodes[:-1], rigid_nodes[1:]):
            graph[first].add(second)
            graph[second].add(first)
    reached, frontier = set(), {0}
    while frontier:
        node = frontier.pop()
        if node in reached:
            continue
        reached.add(node)
        frontier.update(graph[node] - reached)
    connected = len(reached) == len(graph)
    internal_degrees = {
        topology["names"][index]: spring_degree[index]
        for index, kind in enumerate(topology["node_types"])
        if kind == "internal"
    }
    internal_nodes_constrained = all(
        degree >= 3 for degree in internal_degrees.values()
    )
    fixed = np.asarray([
        positions[0, index]
        for index, kind in enumerate(topology["node_types"])
        if kind in ("fixed", "skin1", "skin2")
    ])
    split_skin = any(
        kind in ("skin1", "skin2") for kind in topology["node_types"]
    )
    if split_skin:
        lateral_anchor_banks = (
            np.any(fixed[:, 1] < -0.2) and np.any(fixed[:, 1] > 0.2)
            and np.any(fixed[:, 2] < -0.2) and np.any(fixed[:, 2] > 0.2)
            and np.any(fixed[:, 0] < 0.0) and np.any(fixed[:, 0] > 0.0)
        )
    else:
        lateral_anchor_banks = (
            np.any(fixed[:, 1] < -0.3) and np.any(fixed[:, 1] > 0.3)
            and np.any(fixed[:, 0] < 0.0) and np.any(fixed[:, 0] > 0.0)
        )
    maximum_standoff_length = 0.0
    limb_nodes_supported = True
    rest_positions = topology["local_positions"].detach().cpu().numpy()
    for node_index, kind in enumerate(topology["node_types"]):
        if kind not in ("limb1", "limb2"):
            continue
        point = rest_positions[node_index]
        outer = 0.98 if kind == "limb1" else 1.11
        valid_x = (
            -0.98 <= point[0] <= -0.065
            if kind == "limb1"
            else 0.065 <= point[0] <= 1.11
        )
        scale = abs(point[0]) / outer
        half_y = limb_thickness_scale * (0.035 + 0.030 * scale)
        half_z = limb_thickness_scale * (0.055 + 0.045 * scale)
        transverse_gap = np.hypot(
            max(abs(point[1]) - half_y, 0.0),
            max(abs(point[2]) - half_z, 0.0),
        )
        maximum_standoff_length = max(
            maximum_standoff_length, float(transverse_gap)
        )
        permitted_standoff = (
            skin_radius - 0.05 if skin_radius is not None else 0.15
        )
        limb_nodes_supported &= valid_x and transverse_gap <= permitted_standoff
    finite = bool(np.isfinite(positions).all())
    passed = bool(
        all_nodes_used
        and connected
        and internal_nodes_constrained
        and fixed_to_fixed_springs == 0
        and same_rigid_body_springs == 0
        and lateral_anchor_banks
        and limb_nodes_supported
        and finite
        and bearing_intersections == 0
        and limb_intersections == 0
        and minimum_spring_length > 0.05
        and skin_containment_violations == 0
        and spring_clearance_violations == 0
        and spring_support_violations == 0
    )
    return {
        "passed": passed,
        "joint_axis": topology["data"]["joint_axis"],
        "angles_degrees": [-45.0, 45.0],
        "angle_samples": angle_samples,
        "relaxation_steps": relaxation_steps,
        "rest_length_scale": active_rest_length_scale,
        "nominal_preload_fraction": 1.0 - active_rest_length_scale,
        "all_nodes_used": all_nodes_used,
        "spring_graph_connected": connected,
        "internal_spring_degrees": internal_degrees,
        "all_internal_nodes_have_at_least_three_springs": internal_nodes_constrained,
        "fixed_to_fixed_springs": fixed_to_fixed_springs,
        "same_rigid_body_springs": same_rigid_body_springs,
        "skin_to_skin_springs": skin_to_skin_springs,
        "lateral_anchor_banks": bool(lateral_anchor_banks),
        "split_skin_circumferential_coverage": (
            bool(lateral_anchor_banks) if split_skin else None
        ),
        "all_limb_nodes_rigidly_supported": bool(limb_nodes_supported),
        "maximum_limb_standoff_length_m": maximum_standoff_length,
        "finite_relaxed_geometry": finite,
        "bearing_intersections": bearing_intersections,
        "bearing_intersection_examples": bearing_intersection_examples,
        "bearing_violation_angle_range_degrees": (
            [min(bearing_collision_angles), max(bearing_collision_angles)]
            if bearing_collision_angles else None
        ),
        "worst_bearing_collision_counts": [
            {
                "spring_index": int(index),
                "node_a": topology["names"][int(spring_a[index])],
                "node_b": topology["names"][int(spring_b[index])],
                "collision_states": int(bearing_collision_counts[index]),
            }
            for index in np.argsort(bearing_collision_counts)[::-1][:12]
            if bearing_collision_counts[index] > 0
        ],
        "limb_intersections": limb_intersections,
        "limb_intersection_examples": limb_intersection_examples,
        "minimum_bearing_centerline_radius_m": minimum_bearing_radius,
        "bearing_radius_m": radius,
        "required_bearing_centerline_radius_m": bearing_required_centerline_radius,
        "minimum_spring_length_m": minimum_spring_length,
        "skin_radius_m": skin_radius,
        "joint_boot_radius_m": joint_boot_radius,
        "skin_centerline_containment_violations": skin_containment_violations,
        "skin_centerline_containment_examples": skin_containment_examples,
        "required_spring_to_spring_clearance_m": spring_clearance,
        "minimum_spring_to_spring_clearance_m": minimum_spring_to_spring_clearance,
        "spring_to_spring_clearance_violations": spring_clearance_violations,
        "spring_to_spring_clearance_examples": spring_clearance_examples,
        "spring_collision_angle_range_degrees": (
            [min(spring_collision_angles), max(spring_collision_angles)]
            if spring_collision_angles else None
        ),
        "worst_spring_pair_collision_counts": [
            {
                "spring_a": int(pair[0]),
                "spring_b": int(pair[1]),
                "spring_a_endpoints": [
                    topology["names"][int(spring_a[pair[0]])],
                    topology["names"][int(spring_b[pair[0]])],
                ],
                "spring_b_endpoints": [
                    topology["names"][int(spring_a[pair[1]])],
                    topology["names"][int(spring_b[pair[1]])],
                ],
                "collision_states": int(count),
            }
            for pair, count in sorted(
                spring_pair_collision_counts.items(),
                key=lambda item: item[1], reverse=True,
            )[:12]
        ],
        "worst_spring_collision_counts": [
            {"spring_index": int(index), "collision_states": int(spring_collision_counts[index])}
            for index in np.argsort(spring_collision_counts)[::-1][:12]
        ],
        "support_radius_m": support_radius,
        "required_spring_to_support_clearance_m": spring_radius + support_radius,
        "minimum_spring_to_support_clearance_m": minimum_spring_support_clearance,
        "spring_to_support_clearance_violations": spring_support_violations,
        "spring_to_support_clearance_examples": spring_support_examples,
        "scope": (
            "post-relaxation diagnostic only; checks configured spring-radius "
            "clearance against other springs, supports, limbs, and the bearing, "
            "but excludes collision response, fastener design, deformation, "
            "friction, fatigue, and structural stress"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--angle-samples", type=int, default=19)
    parser.add_argument(
        "--rest-length-scale", type=float,
        help="Optional diagnostic override for rest length / initial geometric length.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tables" / "spatial" / "spatial_feasibility_audit.json",
    )
    args = parser.parse_args()
    report = audit(
        args.topology, args.relaxation_steps, args.angle_samples, args.device,
        args.rest_length_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
