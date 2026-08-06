"""Generate matched sparse and dense variants of the optimized 3D topology."""

from copy import deepcopy
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "topologies" / "spatial"
    / "internal_fan_3d_30_spring_optimized.json"
)
OUTPUT_DIR = PROJECT_ROOT / "topologies" / "spatial"


def write_variant(data, name):
    path = OUTPUT_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    base = json.loads(SOURCE.read_text(encoding="utf-8"))

    sparse = deepcopy(base)
    sparse["name"] = "internal_fan_3d_24_spring_sparse"
    sparse["description"] = (
        "Sparse ablation of the feasible surface-routed 3D fan."
    )
    removed_fixed = {
        "right_inner_front",
        "right_inner_back",
        "right_mid_front",
        "right_mid_back",
        "right_outer_front",
        "right_outer_back",
    }
    sparse["springs"] = [
        spring
        for spring in sparse["springs"]
        if not (
            spring["node_a"] in removed_fixed
            and spring["node_b"].startswith("limb2_")
        )
    ]
    used = {
        endpoint
        for spring in sparse["springs"]
        for endpoint in (spring["node_a"], spring["node_b"])
    }
    sparse["nodes"] = [node for node in sparse["nodes"] if node["name"] in used]

    dense = deepcopy(base)
    dense["name"] = "internal_fan_3d_36_spring_dense"
    dense["description"] = (
        "Dense ablation of the feasible surface-routed 3D fan."
    )
    dense["nodes"].extend([
        {"name": "limb2_top_1", "type": "limb2", "position": [0.32, 0.044, 0.068]},
        {"name": "limb2_bottom_1", "type": "limb2", "position": [0.32, -0.044, -0.068]},
        {"name": "limb2_top_4", "type": "limb2", "position": [0.80, -0.057, 0.087]},
        {"name": "limb2_bottom_4", "type": "limb2", "position": [0.80, 0.057, -0.087]},
        {"name": "limb2_top_6", "type": "limb2", "position": [1.08, 0.064, 0.099]},
        {"name": "limb2_bottom_6", "type": "limb2", "position": [1.08, -0.064, -0.099]}
    ])
    dense["springs"].extend([
        {"node_a": "left_inner_front", "node_b": "limb2_top_1", "stiffness_k": 78.0},
        {"node_a": "left_inner_back", "node_b": "limb2_bottom_1", "stiffness_k": 78.0},
        {"node_a": "left_mid_front", "node_b": "limb2_bottom_4", "stiffness_k": 68.0},
        {"node_a": "left_mid_back", "node_b": "limb2_top_4", "stiffness_k": 68.0},
        {"node_a": "left_outer_front", "node_b": "limb2_top_6", "stiffness_k": 58.0},
        {"node_a": "left_outer_back", "node_b": "limb2_bottom_6", "stiffness_k": 58.0}
    ])

    print(write_variant(sparse, sparse["name"]))
    print(write_variant(dense, dense["name"]))

    denser = deepcopy(dense)
    denser["name"] = "internal_fan_3d_42_spring_denser"
    denser["description"] = "42-spring feasible surface-routed ablation."
    denser["nodes"].extend([
        {"name": "limb2_top_2", "type": "limb2", "position": [0.48, 0.048, 0.074]},
        {"name": "limb2_bottom_2", "type": "limb2", "position": [0.48, -0.048, -0.074]},
        {"name": "limb2_top_3", "type": "limb2", "position": [0.64, 0.052, 0.081]},
        {"name": "limb2_bottom_3", "type": "limb2", "position": [0.64, -0.052, -0.081]},
        {"name": "limb2_top_5", "type": "limb2", "position": [0.96, 0.061, 0.094]},
        {"name": "limb2_bottom_5", "type": "limb2", "position": [0.96, -0.061, -0.094]}
    ])
    denser["springs"].extend([
        {"node_a": "left_outer_front", "node_b": "limb2_top_2", "stiffness_k": 62.0},
        {"node_a": "left_outer_back", "node_b": "limb2_bottom_2", "stiffness_k": 62.0},
        {"node_a": "left_inner_front", "node_b": "limb2_top_5", "stiffness_k": 64.0},
        {"node_a": "left_inner_back", "node_b": "limb2_bottom_5", "stiffness_k": 64.0},
        {"node_a": "right_inner_front", "node_b": "limb2_top_3", "stiffness_k": 74.0},
        {"node_a": "right_inner_back", "node_b": "limb2_bottom_3", "stiffness_k": 74.0}
    ])
    print(write_variant(denser, denser["name"]))

    densest = deepcopy(denser)
    densest["name"] = "internal_fan_3d_48_spring_densest"
    densest["description"] = "48-spring feasible surface-routed ablation."
    densest["nodes"].extend([
        {"name": "limb2_top_a", "type": "limb2", "position": [0.24, -0.042, 0.065]},
        {"name": "limb2_bottom_a", "type": "limb2", "position": [0.24, 0.042, -0.065]},
        {"name": "limb2_top_b", "type": "limb2", "position": [0.72, 0.054, 0.084]},
        {"name": "limb2_bottom_b", "type": "limb2", "position": [0.72, -0.054, -0.084]},
        {"name": "limb2_top_c", "type": "limb2", "position": [1.02, 0.063, 0.096]},
        {"name": "limb2_bottom_c", "type": "limb2", "position": [1.02, -0.063, -0.096]}
    ])
    densest["springs"].extend([
        {"node_a": "right_mid_front", "node_b": "limb2_bottom_a", "stiffness_k": 70.0},
        {"node_a": "right_mid_back", "node_b": "limb2_top_a", "stiffness_k": 70.0},
        {"node_a": "right_outer_front", "node_b": "limb2_top_b", "stiffness_k": 60.0},
        {"node_a": "right_outer_back", "node_b": "limb2_bottom_b", "stiffness_k": 60.0},
        {"node_a": "left_inner_front", "node_b": "limb2_top_c", "stiffness_k": 56.0},
        {"node_a": "left_inner_back", "node_b": "limb2_bottom_c", "stiffness_k": 56.0}
    ])
    print(write_variant(densest, densest["name"]))

    spring_54 = deepcopy(densest)
    spring_54["name"] = "internal_fan_3d_54_spring"
    spring_54["description"] = "54-spring feasible surface-routed ablation."
    spring_54["springs"].extend([
        {"node_a": "left_outer_front", "node_b": "limb2_top_b", "stiffness_k": 54.0},
        {"node_a": "left_outer_back", "node_b": "limb2_bottom_b", "stiffness_k": 54.0},
        {"node_a": "right_inner_front", "node_b": "limb2_top_c", "stiffness_k": 66.0},
        {"node_a": "right_inner_back", "node_b": "limb2_bottom_c", "stiffness_k": 66.0},
        {"node_a": "right_mid_front", "node_b": "limb2_top_6", "stiffness_k": 62.0},
        {"node_a": "right_mid_back", "node_b": "limb2_bottom_6", "stiffness_k": 62.0}
    ])
    print(write_variant(spring_54, spring_54["name"]))

    spring_60 = deepcopy(spring_54)
    spring_60["name"] = "internal_fan_3d_60_spring"
    spring_60["description"] = "60-spring feasible surface-routed ablation."
    spring_60["springs"].extend([
        {"node_a": "left_mid_front", "node_b": "limb2_top_c", "stiffness_k": 52.0},
        {"node_a": "left_mid_back", "node_b": "limb2_bottom_c", "stiffness_k": 52.0},
        {"node_a": "right_outer_front", "node_b": "limb2_top_2", "stiffness_k": 58.0},
        {"node_a": "right_outer_back", "node_b": "limb2_bottom_2", "stiffness_k": 58.0},
        {"node_a": "left_inner_front", "node_b": "limb2_top_b", "stiffness_k": 56.0},
        {"node_a": "left_inner_back", "node_b": "limb2_bottom_b", "stiffness_k": 56.0}
    ])
    print(write_variant(spring_60, spring_60["name"]))


if __name__ == "__main__":
    main()
