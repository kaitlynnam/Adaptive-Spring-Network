"""Generate a 48-spring split-skin topology with real internal and joint nodes."""

from pathlib import Path
import argparse
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "topologies" / "spatial" / "internal_fan_3d_48_spring_densest.json"
OUTPUT = PROJECT_ROOT / "topologies" / "spatial" / "hybrid_internal_skin_3d_48_spring.json"
SKIN_RADIUS = 0.74
BOOT_RADIUS = 0.90
INTERNAL_RADIUS = 0.34
JOINT_RADIUS = 0.23


def point(x, angle_degrees, radius):
    angle = np.radians(angle_degrees)
    return [x, radius * np.cos(angle), radius * np.sin(angle)]


def limb_surface_point(x, angle_degrees, limb_type, thickness_scale=1.0):
    """Place an attachment directly on the tapered rectangular limb surface."""
    outer = 0.98 if limb_type == "limb1" else 1.11
    scale = abs(x) / outer
    half_y = thickness_scale * (0.035 + 0.030 * scale)
    half_z = thickness_scale * (0.055 + 0.045 * scale)
    angle = np.radians(angle_degrees)
    direction_y, direction_z = np.cos(angle), np.sin(angle)
    candidates = []
    if abs(direction_y) > 1e-9:
        candidates.append(half_y / abs(direction_y))
    if abs(direction_z) > 1e-9:
        candidates.append(half_z / abs(direction_z))
    radius = min(candidates)
    return [x, radius * direction_y, radius * direction_z]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-radius", type=float, default=INTERNAL_RADIUS)
    parser.add_argument("--joint-radius", type=float, default=JOINT_RADIUS)
    parser.add_argument("--phase-degrees", type=float, default=7.5)
    parser.add_argument("--direct-torque-lanes", action="store_true")
    parser.add_argument("--direct-joint-radius", type=float, default=0.44)
    parser.add_argument("--direct-phase-degrees", type=float, default=22.5)
    parser.add_argument("--direct-lane-count", type=int, default=12)
    parser.add_argument("--limb-thickness-scale", type=float, default=1.0)
    parser.add_argument("--joint-ring-x", type=float, default=0.20)
    parser.add_argument("--outer-joint-ring-x", type=float, default=0.24)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    data = {
        key: source[key]
        for key in (
            "joint_axis", "bearing_radius", "bearing_half_length",
            "bearing_clearance", "bearing_collision_penalty",
            "rest_angle_degrees", "rest_length_scale",
        )
    }
    data.update({
        "name": (
            f"hybrid_internal_skin_3d_{48 + args.direct_lane_count}_spring_torque_authority"
            if args.direct_torque_lanes else
            "hybrid_internal_skin_3d_48_spring"
        ),
        "description": (
            "Collision-spaced hybrid with 12 free internal nodes, fixed split-skin "
            "anchors, rigid bearing-adjacent eyelets, and optional direct torque lanes."
        ),
        "skin_radius": SKIN_RADIUS,
        "joint_boot_radius": BOOT_RADIUS,
        "spring_clearance": 0.032,
        "spring_radius": 0.015,
        "support_radius": 0.003,
        "joint_nodes_on_limb_surface": True,
        "limb_thickness_scale": args.limb_thickness_scale,
        "nodes": [],
        "springs": [],
    })
    stiffness = [spring["stiffness_k"] for spring in source["springs"]]
    proximal_x, distal_x = (-0.82, -0.55, -0.32), (0.32, 0.55, 0.82)

    def add_node(name, kind, position):
        data["nodes"].append({"name": name, "type": kind, "position": position})

    def add_spring(a, b):
        index = len(data["springs"])
        data["springs"].append({
            "node_a": a, "node_b": b, "stiffness_k": stiffness[index % len(stiffness)],
        })

    for index in range(12):
        angle = 30.0 * index
        ring = index % 3
        add_node(f"skin1_{index:02d}", "skin1", point(proximal_x[ring], angle, SKIN_RADIUS))
        add_node(f"skin2_{index:02d}", "skin2", point(distal_x[ring], angle + 15.0, SKIN_RADIUS))
        add_node(
            f"internal_{index:02d}", "internal",
            point(
                -0.06 if index % 2 == 0 else 0.06,
                angle + args.phase_degrees,
                args.internal_radius,
            ),
        )
        moving_body = "limb2" if index % 2 == 0 else "limb1"
        moving_x = args.joint_ring_x if moving_body == "limb2" else -args.joint_ring_x
        add_node(
            f"joint_internal_{index:02d}", moving_body,
            limb_surface_point(
                moving_x, angle + args.phase_degrees, moving_body,
                args.limb_thickness_scale,
            ),
        )
        direct_body = "limb2" if index % 2 == 0 else "limb1"
        direct_x = (
            args.outer_joint_ring_x
            if direct_body == "limb2" else -args.outer_joint_ring_x
        )
        add_node(
            f"joint_direct_{index:02d}", direct_body,
            limb_surface_point(
                direct_x,
                angle + (0.0 if index % 2 == 0 else 15.0),
                direct_body,
                args.limb_thickness_scale,
            ),
        )
    if args.direct_torque_lanes:
        for lane in range(args.direct_lane_count):
            lane_angle = 360.0 * lane / args.direct_lane_count + args.direct_phase_degrees
            skin_type = "skin1" if lane % 2 == 0 else "skin2"
            joint_type = "limb2" if lane % 2 == 0 else "limb1"
            skin_x = -0.32 if skin_type == "skin1" else 0.32
            joint_x = (
                args.outer_joint_ring_x
                if joint_type == "limb2" else -args.outer_joint_ring_x
            )
            add_node(
                f"skin_direct_{lane:02d}", skin_type,
                point(skin_x, lane_angle, SKIN_RADIUS),
            )
            add_node(
                f"joint_torque_{lane:02d}", joint_type,
                limb_surface_point(
                    joint_x, lane_angle, joint_type, args.limb_thickness_scale
                ),
            )

    for index in range(12):
        add_spring(f"skin1_{index:02d}", f"internal_{index:02d}")
    for index in range(12):
        add_spring(f"skin2_{index:02d}", f"internal_{index:02d}")
    for index in range(12):
        add_spring(f"internal_{index:02d}", f"joint_internal_{index:02d}")
    for index in range(12):
        add_spring(f"internal_{index:02d}", f"joint_direct_{index:02d}")
    if args.direct_torque_lanes:
        for lane in range(args.direct_lane_count):
            add_spring(f"skin_direct_{lane:02d}", f"joint_torque_{lane:02d}")

    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
