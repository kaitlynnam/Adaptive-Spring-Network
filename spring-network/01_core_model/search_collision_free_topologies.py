"""Search collision-free 60-spring layouts for passive torque authority."""

from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from audit_spatial_feasibility import audit  # noqa: E402
from mechanics_3d import load_spatial_topology, torque_components_and_residual  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "topologies" / "spatial" / "collision_free_search"
TABLE_PATH = PROJECT_ROOT / "tables" / "spatial" / "collision_free_topology_search.csv"


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


def build_candidate(parameters, candidate_index, lanes=12):
    nodes, springs = [], []

    def add_node(name, kind, position):
        nodes.append({"name": name, "type": kind, "position": position})

    def add_spring(name, node_a, node_b, index):
        stiffness = 70.0 + 50.0 * ((index * 7) % 13) / 12.0
        springs.append({
            "name": name, "node_a": node_a, "node_b": node_b,
            "stiffness_k": float(stiffness),
        })

    spring_index = 0
    for lane in range(lanes):
        base_angle = 2.0 * np.pi * lane / lanes + parameters["phase"]
        distal_angle = base_angle + parameters["distal_offset"]
        internal_angle = base_angle + parameters["internal_offset"]
        by, bz = np.cos(base_angle), np.sin(base_angle)
        dy, dz = np.cos(distal_angle), np.sin(distal_angle)
        iy, iz = np.cos(internal_angle), np.sin(internal_angle)
        names = {
            key: f"{key}_{lane:02d}"
            for key in ("skin1", "limb1", "internal", "limb2", "skin2")
        }
        skin_x = parameters["skin_x"]
        limb_x = parameters["limb_x"]
        radius = parameters["internal_radius"]
        add_node(names["skin1"], "skin1", [-skin_x, 0.74 * by, 0.74 * bz])
        add_node(names["limb1"], "limb1", limb_surface_point(-limb_x, base_angle, "limb1"))
        add_node(
            names["internal"], "internal",
            [parameters["internal_x"], radius * iy, radius * iz],
        )
        add_node(names["limb2"], "limb2", limb_surface_point(limb_x, distal_angle, "limb2"))
        add_node(names["skin2"], "skin2", [skin_x, 0.74 * dy, 0.74 * dz])
        connections = (
            ("skin1_internal", names["skin1"], names["internal"]),
            ("limb1_internal", names["limb1"], names["internal"]),
            ("internal_limb2", names["internal"], names["limb2"]),
            ("internal_skin2", names["internal"], names["skin2"]),
            ("outer_cross", names["skin1"], names["skin2"]),
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
        "rest_length_scale": 0.78,
        "name": f"collision_free_{5 * lanes}s_authority_candidate_{candidate_index:04d}",
        "description": (
            f"{lanes} routed lanes with internal support and outer "
            "cross-joint torque springs."
        ),
        "skin_radius": 0.74,
        "joint_boot_radius": 0.9,
        "spring_clearance": 0.012,
        "spring_radius": 0.005,
        "support_radius": 0.003,
        "joint_nodes_on_limb_surface": True,
        "search_parameters": parameters,
        "nodes": nodes,
        "springs": springs,
    }


def authority_metrics(path, device, relaxation_steps):
    topology = load_spatial_topology(path, device)
    theta = torch.linspace(-np.pi / 4, np.pi / 4, 13, device=device)
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(theta), 1)
    components, residual, _ = torque_components_and_residual(
        topology, theta, stiffness, relaxation_steps
    )
    authority = torch.sum(torch.abs(components), dim=1)
    nonzero = torch.max(torch.abs(components), dim=0).values > 1e-6
    score = torch.mean(authority) - 0.35 * torch.std(authority)
    return {
        "authority_score": float(score),
        "mean_authority_nm": float(torch.mean(authority)),
        "minimum_authority_nm": float(torch.min(authority)),
        "nonzero_direct_torque_springs": int(torch.sum(nonzero)),
        "mean_force_residual_n": float(torch.mean(residual)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--relaxation-steps", type=int, default=120)
    parser.add_argument("--angle-samples", type=int, default=9)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=12)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    retained = []
    with tempfile.TemporaryDirectory(prefix="spring_topology_search_") as directory:
        candidate_path = Path(directory) / "candidate.json"
        for index in range(args.candidates):
            parameters = {
                "phase": float(rng.uniform(0.0, np.pi / 6.0)),
                "distal_offset": float(rng.uniform(-0.12, 0.12)),
                "internal_offset": float(rng.uniform(-0.10, 0.10)),
                # Keep cross-joint skin anchors inside the flexible boot so the
                # straight outer torque lane remains contained while bending.
                "skin_x": float(rng.uniform(0.18, 0.34)),
                "limb_x": float(rng.uniform(0.30, 0.70)),
                "internal_x": float(rng.uniform(-0.10, 0.10)),
                "internal_radius": float(rng.uniform(0.30, 0.62)),
            }
            candidate = build_candidate(parameters, index, args.lanes)
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            report = audit(
                candidate_path, args.relaxation_steps, args.angle_samples,
                args.device,
            )
            row = {
                "candidate": index,
                "feasible": report["passed"],
                "minimum_spring_clearance_m": report["minimum_spring_to_spring_clearance_m"],
                "minimum_bearing_radius_m": report["minimum_bearing_centerline_radius_m"],
                "spring_clearance_violations": report["spring_to_spring_clearance_violations"],
                "bearing_intersections": report["bearing_intersections"],
                "limb_intersections": report["limb_intersections"],
                "skin_containment_violations": report["skin_centerline_containment_violations"],
                "spring_support_violations": report["spring_to_support_clearance_violations"],
                **parameters,
            }
            if report["passed"]:
                metrics = authority_metrics(candidate_path, args.device, args.relaxation_steps)
                row.update(metrics)
                retained.append((metrics["authority_score"], index, candidate, row))
            rows.append(row)
            print(
                f"{index + 1:3d}/{args.candidates}: feasible={report['passed']} "
                f"authority={row.get('authority_score', '')}", flush=True,
            )

    retained.sort(key=lambda item: item[0], reverse=True)
    output_dir = OUTPUT_DIR / f"{5 * args.lanes:03d}_springs"
    table_path = TABLE_PATH.with_name(
        f"collision_free_topology_search_{5 * args.lanes:03d}_springs.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, index, candidate, _) in enumerate(retained[:args.keep], start=1):
        path = output_dir / f"rank_{rank:02d}_candidate_{index:04d}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"feasible={len(retained)}/{args.candidates}")
    for rank, (score, index, _, row) in enumerate(retained[:args.keep], start=1):
        print(
            f"rank={rank} candidate={index} authority={score:.4f} "
            f"minimum_clearance={row['minimum_spring_clearance_m']:.4f}"
        )
    print(table_path)


if __name__ == "__main__":
    main()
