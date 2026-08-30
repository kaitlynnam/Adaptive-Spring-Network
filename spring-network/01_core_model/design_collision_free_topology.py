"""Generate a collision-routed 60-spring split-skin topology candidate."""

from pathlib import Path
import argparse
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "topologies" / "spatial"
    / "split_skin_collision_free_3d_60_spring.json"
)


def limb_surface_point(x, angle, limb_type):
    outer = 0.98 if limb_type == "limb1" else 1.11
    scale = abs(x) / outer
    half_y = 0.035 + 0.030 * scale
    half_z = 0.055 + 0.045 * scale
    dy, dz = np.cos(angle), np.sin(angle)
    radius = min(
        half_y / max(abs(dy), 1e-9),
        half_z / max(abs(dz), 1e-9),
    )
    return [float(x), float(radius * dy), float(radius * dz)]


def build_topology(lanes=12):
    if lanes != 12:
        raise ValueError("This paper candidate is defined for exactly 12 lanes")
    nodes, springs = [], []

    def add_node(name, kind, position):
        nodes.append({"name": name, "type": kind, "position": position})

    def add_spring(name, node_a, node_b, index):
        # Preserve heterogeneous baseline stiffness without random generation.
        stiffness = 70.0 + 50.0 * ((index * 7) % 13) / 12.0
        springs.append({
            "name": name,
            "node_a": node_a,
            "node_b": node_b,
            "stiffness_k": float(stiffness),
        })

    spring_index = 0
    for lane in range(lanes):
        angle = 2.0 * np.pi * lane / lanes + np.radians(7.5)
        y, z = float(np.cos(angle)), float(np.sin(angle))
        names = {
            "skin1": f"skin1_{lane:02d}",
            "limb1": f"limb1_{lane:02d}",
            "internal1": f"internal_proximal_{lane:02d}",
            "internal2": f"internal_distal_{lane:02d}",
            "limb2": f"limb2_{lane:02d}",
            "skin2": f"skin2_{lane:02d}",
        }
        add_node(names["skin1"], "skin1", [-0.82, 0.74 * y, 0.74 * z])
        add_node(names["limb1"], "limb1", limb_surface_point(-0.46, angle, "limb1"))
        add_node(names["internal1"], "internal", [-0.14, 0.40 * y, 0.40 * z])
        add_node(names["internal2"], "internal", [0.14, 0.40 * y, 0.40 * z])
        add_node(names["limb2"], "limb2", limb_surface_point(0.46, angle, "limb2"))
        add_node(names["skin2"], "skin2", [0.82, 0.74 * y, 0.74 * z])
        connections = (
            ("skin1_internal", names["skin1"], names["internal1"]),
            ("limb1_internal", names["limb1"], names["internal1"]),
            ("internal_bridge", names["internal1"], names["internal2"]),
            ("internal_limb2", names["internal2"], names["limb2"]),
            ("internal_skin2", names["internal2"], names["skin2"]),
        )
        for label, node_a, node_b in connections:
            add_spring(f"{label}_{lane:02d}", node_a, node_b, spring_index)
            spring_index += 1

    return {
        "joint_axis": [0.0, 1.0, 0.0],
        "bearing_radius": 0.12,
        "bearing_half_length": 0.13,
        "bearing_clearance": 0.02,
        "bearing_collision_penalty": 0.0,
        "rest_angle_degrees": 0.0,
        "rest_length_scale": 0.75,
        "name": "split_skin_collision_free_3d_60_spring",
        "description": (
            "Twelve separated five-spring lanes routed through paired free "
            "internal nodes outside the revolute bearing."
        ),
        "skin_radius": 0.74,
        "joint_boot_radius": 0.9,
        "spring_clearance": 0.012,
        "spring_radius": 0.005,
        "support_radius": 0.003,
        "joint_nodes_on_limb_surface": True,
        "nodes": nodes,
        "springs": springs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    topology = build_topology()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
