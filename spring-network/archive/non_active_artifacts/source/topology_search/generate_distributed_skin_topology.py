"""Generate a collision-spaced 48-spring split-skin joint candidate."""

from pathlib import Path
import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "topologies" / "spatial"
    / "internal_fan_3d_48_spring_densest.json"
)
OUTPUT = (
    PROJECT_ROOT / "topologies" / "spatial"
    / "distributed_skin_3d_48_spring.json"
)
SKIN_RADIUS = 0.74
JOINT_FRAME_RADIUS = 0.44
JOINT_BOOT_RADIUS = 0.90


def position(x, angle_degrees, radius):
    angle = np.radians(angle_degrees)
    return [x, radius * np.cos(angle), radius * np.sin(angle)]


def main():
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
        "name": "distributed_skin_3d_48_spring",
        "description": (
            "48 independent circumferential spring lanes between fixed split-skin "
            "anchors and opposite-body joint-frame eyelets."
        ),
        "skin_radius": SKIN_RADIUS,
        "joint_boot_radius": JOINT_BOOT_RADIUS,
        "spring_clearance": 0.032,
        "nodes": [],
        "springs": [],
    })
    stiffness = [spring["stiffness_k"] for spring in source["springs"]]
    # Three axial rings prevent all 48 springs from occupying one joint section.
    proximal_x = (-0.82, -0.55, -0.30)
    distal_x = (0.30, 0.55, 0.82)
    for index in range(24):
        ring = index % 3
        angle = 15.0 * index
        skin_name = f"skin1_anchor_{index:02d}"
        joint_name = f"joint2_eyelet_{index:02d}"
        data["nodes"].extend([
            {
                "name": skin_name,
                "type": "skin1",
                "position": position(proximal_x[ring], angle, SKIN_RADIUS),
            },
            {
                "name": joint_name,
                "type": "limb2",
                "position": position(distal_x[ring], angle, JOINT_FRAME_RADIUS),
            },
        ])
        data["springs"].append({
            "node_a": skin_name,
            "node_b": joint_name,
            "stiffness_k": stiffness[index],
        })
    for index in range(24):
        ring = index % 3
        angle = 15.0 * index + 7.5
        skin_name = f"skin2_anchor_{index:02d}"
        joint_name = f"joint1_eyelet_{index:02d}"
        data["nodes"].extend([
            {
                "name": skin_name,
                "type": "skin2",
                "position": position(distal_x[ring], angle, SKIN_RADIUS),
            },
            {
                "name": joint_name,
                "type": "limb1",
                "position": position(proximal_x[ring], angle, JOINT_FRAME_RADIUS),
            },
        ])
        data["springs"].append({
            "node_a": skin_name,
            "node_b": joint_name,
            "stiffness_k": stiffness[index + 24],
        })
    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
