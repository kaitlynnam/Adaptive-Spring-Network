"""Search spatial lane count, connectivity geometry, and continuous node positions."""

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from audit_spatial_feasibility import audit
from mechanics_3d import load_spatial_topology, torque_components_and_residual


OUTPUT_DIR = PROJECT_ROOT / "topologies" / "spatial" / "global_search"
TABLE_PATH = PROJECT_ROOT / "tables" / "spatial" / "global_topology_search.csv"


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
    return [x, radius * dy, radius * dz]


def build_candidate(lanes, rng, index):
    nodes, springs = [], []

    def node(name, kind, position):
        nodes.append({"name": name, "type": kind, "position": position})

    def spring(name, a, b):
        springs.append({
            "name": name, "node_a": a, "node_b": b,
            "stiffness_k": float(rng.uniform(60.0, 115.0)),
        })

    phase = rng.uniform(0.0, 2.0 * np.pi)
    spacing = 2.0 * np.pi / lanes
    for lane in range(lanes):
        base = phase + lane * spacing
        skin1_angle = base + rng.uniform(-0.10, 0.10)
        skin2_angle = base + 0.45 * spacing + rng.uniform(-0.10, 0.10)
        internal_angle = base + 0.22 * spacing + rng.uniform(-0.08, 0.08)
        skin1_x = -rng.uniform(0.30, 0.92)
        skin2_x = rng.uniform(0.30, 0.92)
        internal_x = rng.uniform(-0.14, 0.14)
        internal_radius = rng.uniform(0.28, 0.48)
        node(
            f"skin1_{lane:02d}", "skin1",
            [skin1_x, 0.74 * np.cos(skin1_angle), 0.74 * np.sin(skin1_angle)],
        )
        node(
            f"skin2_{lane:02d}", "skin2",
            [skin2_x, 0.74 * np.cos(skin2_angle), 0.74 * np.sin(skin2_angle)],
        )
        node(
            f"internal_{lane:02d}", "internal",
            [
                internal_x,
                internal_radius * np.cos(internal_angle),
                internal_radius * np.sin(internal_angle),
            ],
        )
        limb_type = "limb2" if lane % 2 == 0 else "limb1"
        sign = 1.0 if limb_type == "limb2" else -1.0
        first_x = sign * rng.uniform(0.25, 0.62)
        second_x = sign * rng.uniform(0.25, 0.68)
        first_angle = internal_angle + rng.uniform(-0.10, 0.10)
        second_angle = base + rng.uniform(-0.10, 0.10)
        node(
            f"joint_a_{lane:02d}", limb_type,
            limb_surface_point(first_x, first_angle, limb_type),
        )
        node(
            f"joint_b_{lane:02d}", limb_type,
            limb_surface_point(second_x, second_angle, limb_type),
        )
        spring(f"skin1_internal_{lane:02d}", f"skin1_{lane:02d}", f"internal_{lane:02d}")
        spring(f"skin2_internal_{lane:02d}", f"skin2_{lane:02d}", f"internal_{lane:02d}")
        spring(f"internal_joint_a_{lane:02d}", f"internal_{lane:02d}", f"joint_a_{lane:02d}")
        spring(f"internal_joint_b_{lane:02d}", f"internal_{lane:02d}", f"joint_b_{lane:02d}")
    return {
        "name": f"global_surface_{4 * lanes:02d}s_candidate_{index:04d}",
        "description": "Continuous-position lane-count candidate with direct surface mounts.",
        "joint_axis": [0.0, 1.0, 0.0],
        "bearing_radius": 0.12,
        "bearing_half_length": 0.10,
        "bearing_clearance": 0.02,
        "bearing_collision_penalty": 500000.0,
        "rest_angle_degrees": 0.0,
        "rest_length_scale": 0.78,
        "skin_radius": 0.74,
        "joint_boot_radius": 0.9,
        "spring_clearance": 0.05,
        "spring_radius": 0.015,
        "support_radius": 0.003,
        "joint_nodes_on_limb_surface": True,
        "limb_thickness_scale": 1.0,
        "nodes": nodes,
        "springs": springs,
    }


def authority_score(path, device, relaxation_steps):
    topology = load_spatial_topology(path, device)
    theta = torch.linspace(-np.pi / 4, np.pi / 4, 13, device=device)
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(theta), 1)
    components, residual, _ = torque_components_and_residual(
        topology, theta, stiffness, relaxation_steps
    )
    authority = torch.sum(torch.abs(components), dim=1)
    score = torch.mean(authority) - 0.35 * torch.std(authority)
    return float(score), float(torch.mean(residual)), float(torch.max(residual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spring-counts", default="24,32,40,48,56,64")
    parser.add_argument("--candidates-per-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--screen-relaxation-steps", type=int, default=120)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required but unavailable")
    counts = [int(item) for item in args.spring_counts.split(",")]
    if any(count % 4 for count in counts):
        raise SystemExit("Each spring count must be divisible by four")
    rng = np.random.default_rng(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidate_index = 0
    for spring_count in counts:
        lanes = spring_count // 4
        for local_index in range(args.candidates_per_count):
            candidate = build_candidate(lanes, rng, candidate_index)
            path = OUTPUT_DIR / f"candidate_{candidate_index:04d}_{spring_count:02d}s.json"
            path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
            report = audit(
                path, relaxation_steps=args.screen_relaxation_steps,
                angle_samples=9, device=args.device,
            )
            row = {
                "candidate": candidate_index,
                "spring_count": spring_count,
                "quick_feasible": report["passed"],
                "minimum_spring_spacing_m":
                    report["minimum_spring_to_spring_clearance_m"],
                "authority_score": "",
                "residual_mean_n": "",
                "residual_max_n": "",
                "path": str(path),
            }
            if report["passed"]:
                score, mean_residual, max_residual = authority_score(
                    path, args.device, args.screen_relaxation_steps
                )
                row.update({
                    "authority_score": score,
                    "residual_mean_n": mean_residual,
                    "residual_max_n": max_residual,
                })
            rows.append(row)
            candidate_index += 1
            print(
                f"{spring_count:2d}s {local_index + 1:2d}/"
                f"{args.candidates_per_count}: feasible={report['passed']} "
                f"score={row['authority_score']}"
            )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for count in counts:
        feasible = [
            row for row in rows
            if row["spring_count"] == count and row["quick_feasible"]
        ]
        feasible.sort(key=lambda row: float(row["authority_score"]), reverse=True)
        best = feasible[0] if feasible else None
        print(
            f"{count:2d} springs: {len(feasible)}/{args.candidates_per_count} feasible"
            + (
                f", best candidate {best['candidate']} "
                f"score {float(best['authority_score']):.4f}"
                if best else ""
            )
        )
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
