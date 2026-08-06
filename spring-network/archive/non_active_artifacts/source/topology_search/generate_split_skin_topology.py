"""Create a 48-spring candidate with anchors on split robot-skin cylinders."""

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
    / "internal_fan_3d_48_spring_split_skin.json"
)


SKIN_RADIUS = 0.46


def cylinder_position(x, angle_degrees, radius=SKIN_RADIUS):
    angle = np.radians(angle_degrees)
    return [x, radius * np.cos(angle), radius * np.sin(angle)]


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    data["name"] = "internal_fan_3d_48_spring_split_skin"
    data["description"] = (
        "48-spring candidate enclosed by enlarged proximal and distal robot-skin "
        "cylinders. Skin anchors connect only to internal or joint nodes."
    )
    data["skin_radius"] = SKIN_RADIUS
    # Six anchors per half give genuine front, back, top, bottom, and diagonal
    # coverage. Inner-ring anchors carry most cross-joint paths; outer anchors
    # remain useful for the free-node fan.
    proximal = {
        "left_outer_front": (-0.82, 45),
        "left_outer_back": (-0.82, 225),
        "left_mid_front": (-0.52, 90),
        "left_mid_back": (-0.52, 270),
        "left_inner_front": (-0.30, 0),
        "left_inner_back": (-0.30, 180),
    }
    distal = {
        "right_inner_front": (0.30, 0),
        "right_inner_back": (0.30, 180),
        "right_mid_front": (0.52, 90),
        "right_mid_back": (0.52, 270),
        "right_outer_front": (0.82, 45),
        "right_outer_back": (0.82, 225),
    }
    for node in data["nodes"]:
        if node["name"] in proximal:
            node["type"] = "skin1"
            node["position"] = cylinder_position(*proximal[node["name"]])
        elif node["name"] in distal:
            node["type"] = "skin2"
            node["position"] = cylinder_position(*distal[node["name"]])
        elif node["type"] in ("limb1", "limb2"):
            # Spread rigid joint eyelets around an internal mounting frame. They
            # remain joint/limb nodes, distinct from the outer fixed skin anchors.
            point = np.asarray(node["position"], dtype=float)
            transverse = point[1:]
            norm = np.linalg.norm(transverse)
            if norm < 1e-9:
                transverse, norm = np.array([1.0, 0.0]), 1.0
            point[1:] = 0.24 * transverse / norm
            node["position"] = point.tolist()

    node_type = {node["name"]: node["type"] for node in data["nodes"]}
    node_position = {
        node["name"]: np.asarray(node["position"], dtype=float)
        for node in data["nodes"]
    }
    base_anchors = {"skin1": proximal, "skin2": distal}
    existing = set()
    for spring in data["springs"]:
        a, b = spring["node_a"], spring["node_b"]
        body = {
            "skin1": "proximal", "limb1": "proximal",
            "skin2": "distal", "limb2": "distal",
        }
        if body.get(node_type[a]) is not None and body.get(node_type[a]) == body.get(node_type[b]):
            # A rigid-body-only spring is useless. Move its original base
            # anchor to the other skin, keeping the closest circumferential side.
            base_name, other_name = (a, b) if a in (*proximal, *distal) else (b, a)
            opposite = (
                "skin2" if body[node_type[other_name]] == "proximal" else "skin1"
            )
            other = node_position[other_name]
            other_angle = np.degrees(np.arctan2(other[2], other[1])) % 360.0
            replacement = min(
                base_anchors[opposite],
                key=lambda name: abs(
                    (base_anchors[opposite][name][1] - other_angle + 180.0)
                    % 360.0 - 180.0
                ),
            )
            if spring["node_a"] == base_name:
                spring["node_a"] = replacement
            else:
                spring["node_b"] = replacement
        a, b = spring["node_a"], spring["node_b"]
        type_a, type_b = node_type[a], node_type[b]
        if type_a.startswith("skin") and type_b in ("limb1", "limb2"):
            prefix = "left" if type_a == "skin1" else "right"
            spring["node_a"] = f"{prefix}_inner_{'front' if node_position[b][1] >= 0 else 'back'}"
        elif type_b.startswith("skin") and type_a in ("limb1", "limb2"):
            prefix = "left" if type_b == "skin1" else "right"
            spring["node_b"] = f"{prefix}_inner_{'front' if node_position[a][1] >= 0 else 'back'}"
        endpoint_types = (
            node_type[spring["node_a"]], node_type[spring["node_b"]]
        )
        if endpoint_types[0].startswith("skin") and endpoint_types[1].startswith("skin"):
            raise ValueError("Skin anchor cannot connect directly to another skin anchor")
        edge = tuple(sorted((spring["node_a"], spring["node_b"])))
        if edge in existing:
            raise ValueError(f"Duplicate spring edge generated: {edge}")
        existing.add(edge)

    OUTPUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
