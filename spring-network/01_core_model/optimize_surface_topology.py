"""Randomized geometry search for a surface-mounted spatial spring topology."""

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


DEFAULT_BASE = (
    PROJECT_ROOT / "topologies" / "spatial"
    / "hybrid_internal_skin_48_surface_feasible.json"
)
OUTPUT_DIR = PROJECT_ROOT / "topologies" / "spatial" / "surface_search"
TABLE_PATH = PROJECT_ROOT / "tables" / "spatial" / "surface_topology_search.csv"


def limb_surface_point(x, angle, limb_type, thickness_scale=1.0):
    outer = 0.98 if limb_type == "limb1" else 1.11
    axial_scale = abs(x) / outer
    half_y = thickness_scale * (0.035 + 0.030 * axial_scale)
    half_z = thickness_scale * (0.055 + 0.045 * axial_scale)
    dy, dz = np.cos(angle), np.sin(angle)
    limits = []
    if abs(dy) > 1e-9:
        limits.append(half_y / abs(dy))
    if abs(dz) > 1e-9:
        limits.append(half_z / abs(dz))
    radius = min(limits)
    return [x, radius * dy, radius * dz]


def mutate(base, rng, candidate_index):
    data = json.loads(json.dumps(base))
    data["name"] = f"surface_search_candidate_{candidate_index:03d}"
    thickness = float(data.get("limb_thickness_scale", 1.0))
    for node in data["nodes"]:
        kind = node["type"]
        point = np.asarray(node["position"], dtype=float)
        angle = np.arctan2(point[2], point[1])
        if kind in ("limb1", "limb2"):
            magnitude = rng.uniform(0.24, 0.68)
            x = -magnitude if kind == "limb1" else magnitude
            angle += rng.uniform(-np.radians(11.0), np.radians(11.0))
            node["position"] = limb_surface_point(x, angle, kind, thickness)
        elif kind in ("skin1", "skin2"):
            magnitude = rng.uniform(0.28, 0.92)
            point[0] = -magnitude if kind == "skin1" else magnitude
            angle += rng.uniform(-np.radians(8.0), np.radians(8.0))
            radius = float(data["skin_radius"])
            point[1], point[2] = radius * np.cos(angle), radius * np.sin(angle)
            node["position"] = point.tolist()
        elif kind == "internal":
            point[0] = rng.uniform(-0.14, 0.14)
            radius = rng.uniform(0.27, 0.50)
            angle += rng.uniform(-np.radians(10.0), np.radians(10.0))
            point[1], point[2] = radius * np.cos(angle), radius * np.sin(angle)
            node["position"] = point.tolist()
    return data


def authority_score(path, device):
    topology = load_spatial_topology(path, device)
    theta = torch.linspace(-np.pi / 4, np.pi / 4, 13, device=device)
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(theta), 1)
    components, residual, _ = torque_components_and_residual(
        topology, theta, stiffness, relaxation_steps=160
    )
    # Sum of independently adjustable torque magnitudes, balanced across angles.
    authority = torch.sum(torch.abs(components), dim=1)
    score = float(torch.mean(authority) - 0.35 * torch.std(authority))
    return score, float(torch.mean(residual)), float(torch.max(residual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--candidates", type=int, default=36)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.candidates):
        candidate = mutate(base, rng, index)
        path = OUTPUT_DIR / f"candidate_{index:03d}.json"
        path.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        report = audit(path, relaxation_steps=120, angle_samples=9, device=args.device)
        row = {
            "candidate": index,
            "quick_feasible": report["passed"],
            "minimum_spring_clearance_m": report["minimum_spring_to_spring_clearance_m"],
            "minimum_bearing_radius_m": report["minimum_bearing_centerline_radius_m"],
            "authority_score": "",
            "residual_mean_n": "",
            "residual_max_n": "",
            "path": str(path),
        }
        if report["passed"]:
            score, mean_residual, max_residual = authority_score(path, args.device)
            row.update({
                "authority_score": score,
                "residual_mean_n": mean_residual,
                "residual_max_n": max_residual,
            })
        rows.append(row)
        print(
            f"{index + 1:3d}/{args.candidates}: feasible={report['passed']} "
            f"score={row['authority_score']}"
        )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    feasible = [row for row in rows if row["quick_feasible"]]
    feasible.sort(key=lambda row: float(row["authority_score"]), reverse=True)
    print(f"feasible={len(feasible)}/{len(rows)}")
    for row in feasible[:5]:
        print(
            f"candidate={row['candidate']} score={float(row['authority_score']):.5f} "
            f"clearance={float(row['minimum_spring_clearance_m']):.5f}"
        )
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
