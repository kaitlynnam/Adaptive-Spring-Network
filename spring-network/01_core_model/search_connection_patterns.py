"""Search collision-free rewiring patterns using exact relaxed torque bases."""

from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile

import numpy as np
from scipy.optimize import lsq_linear


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT_ROOT / "01_core_model"),
    str(PROJECT_ROOT / "04_adaptive_learning"),
]

from audit_spatial_feasibility import audit  # noqa: E402
from mechanics_3d import load_spatial_topology  # noqa: E402
from period_adaptive_support import spatial_initial_basis  # noqa: E402
from profile_generator import generate_profile_parameters, profile_torque  # noqa: E402


DEFAULT_SOURCE = (
    PROJECT_ROOT / "topologies" / "spatial" / "collision_free_refined" / "rank_03.json"
)
OUTPUT_DIR = PROJECT_ROOT / "topologies" / "spatial" / "connection_pattern_search"
TABLE_PATH = PROJECT_ROOT / "tables" / "spatial" / "connection_pattern_search.csv"


def lane_number(name):
    return int(name.rsplit("_", 1)[1])


def shifted(name, offset, lanes):
    prefix = name.rsplit("_", 1)[0]
    return f"{prefix}_{(lane_number(name) + offset) % lanes:02d}"


def rewire(source, outer_shift, distal_route_shift):
    data = json.loads(source.read_text(encoding="utf-8"))
    lanes = sum(node["type"] == "skin1" for node in data["nodes"])
    positions = {
        node["name"]: np.asarray(node["position"], dtype=float)
        for node in data["nodes"]
    }
    for spring in data["springs"]:
        if spring["name"].startswith("outer_cross"):
            spring["node_b"] = shifted(spring["node_b"], outer_shift, lanes)
        elif spring["name"].startswith(("internal_limb2", "internal_skin2")):
            spring["node_b"] = shifted(spring["node_b"], distal_route_shift, lanes)
        # Rewiring changes geometry, so always reconstruct the intended rest length.
        geometric = np.linalg.norm(
            positions[spring["node_b"]] - positions[spring["node_a"]]
        )
        if spring["name"].startswith("outer_cross"):
            scale = data.get("refinement", {}).get("cross_rest_scale", 1.0)
            spring["rest_length"] = float(scale * geometric)
        else:
            spring.pop("rest_length", None)
    data["name"] = f"connection_pattern_outer{outer_shift:+d}_route{distal_route_shift:+d}"
    data["description"] = (
        "Collision-screened rotationally staggered connection pattern; "
        f"outer shift {outer_shift:+d}, distal routed shift {distal_route_shift:+d}."
    )
    data["connection_pattern"] = {
        "outer_lane_shift": outer_shift,
        "distal_route_lane_shift": distal_route_shift,
    }
    return data


def exact_fit_metrics(path, angles, targets, steps, device):
    topology = load_spatial_topology(path, device)
    basis = spatial_initial_basis(topology, angles, steps)
    base = topology["initial_stiffness"].detach().cpu().numpy()
    lower, upper = base * 10.0 ** -0.3, base * 100.0
    rmses = []
    for target in targets:
        result = lsq_linear(basis, target, bounds=(lower, upper), max_iter=250)
        rmses.append(np.sqrt(np.mean((target - basis @ result.x) ** 2)))
    return float(np.mean(rmses)), float(np.percentile(rmses, 90))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--relaxation-steps", type=int, default=120)
    parser.add_argument("--profiles", type=int, default=40)
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--max-shift", type=int, default=3)
    args = parser.parse_args()

    angles = np.radians(np.arange(-45.0, 46.0, 5.0))
    profiles = generate_profile_parameters(np.random.default_rng(20260814), args.profiles)
    targets = np.stack([profile_torque(angles, profile) for profile in profiles])
    rows, retained = [], []
    shifts = range(-args.max_shift, args.max_shift + 1)
    with tempfile.TemporaryDirectory(prefix="connection_pattern_search_") as directory:
        path = Path(directory) / "candidate.json"
        for outer_shift in shifts:
            for route_shift in shifts:
                data = rewire(args.source, outer_shift, route_shift)
                path.write_text(json.dumps(data), encoding="utf-8")
                report = audit(path, args.relaxation_steps, 13, args.device)
                row = {
                    "outer_lane_shift": outer_shift,
                    "distal_route_lane_shift": route_shift,
                    "collision_audit": "pass" if report["passed"] else "fail",
                    "minimum_spring_clearance_m": report["minimum_spring_to_spring_clearance_m"],
                    "spring_clearance_violations": report["spring_to_spring_clearance_violations"],
                    "bearing_intersections": report["bearing_intersections"],
                    "limb_intersections": report["limb_intersections"],
                }
                if report["passed"]:
                    mean_rmse, p90_rmse = exact_fit_metrics(
                        path, angles, targets, args.relaxation_steps, args.device
                    )
                    row.update({
                        "exact_bounded_fit_mean_rmse_nm": mean_rmse,
                        "exact_bounded_fit_p90_rmse_nm": p90_rmse,
                    })
                    retained.append((mean_rmse, data, row))
                rows.append(row)
                print(
                    f"outer={outer_shift:+d} route={route_shift:+d} "
                    f"audit={row['collision_audit']} "
                    f"fit={row.get('exact_bounded_fit_mean_rmse_nm', '')}",
                    flush=True,
                )

    retained.sort(key=lambda item: item[0])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank, (_, data, _) in enumerate(retained[:args.keep], 1):
        (OUTPUT_DIR / f"rank_{rank:02d}.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for rank, (score, _, row) in enumerate(retained[:args.keep], 1):
        print(
            f"rank={rank} fit={score:.3f} outer={row['outer_lane_shift']:+d} "
            f"route={row['distal_route_lane_shift']:+d} "
            f"clearance={row['minimum_spring_clearance_m']:.4f}"
        )
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
