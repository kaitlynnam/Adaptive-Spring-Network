"""Refine collision-free candidates for bounded torque fit, not raw authority."""

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

from mechanics_3d import load_spatial_topology  # noqa: E402
from period_adaptive_support import spatial_initial_basis  # noqa: E402
from profile_generator import generate_profile_parameters, profile_torque  # noqa: E402


SOURCE_DIR = PROJECT_ROOT / "topologies" / "spatial" / "collision_free_search"
OUTPUT_DIR = PROJECT_ROOT / "topologies" / "spatial" / "collision_free_refined"
TABLE_PATH = PROJECT_ROOT / "tables" / "spatial" / "collision_free_refinement.csv"


def configure(source, cross_stiffness, cross_rest_scale, internal_stiffness):
    data = json.loads(source.read_text(encoding="utf-8"))
    positions = {
        node["name"]: np.asarray(node["position"], dtype=float)
        for node in data["nodes"]
    }
    for spring in data["springs"]:
        is_cross = spring["name"].startswith("outer_cross")
        spring["stiffness_k"] = float(
            cross_stiffness if is_cross else internal_stiffness
        )
        if is_cross:
            geometric = np.linalg.norm(
                positions[spring["node_b"]] - positions[spring["node_a"]]
            )
            spring["rest_length"] = float(cross_rest_scale * geometric)
    data["name"] = (
        f"{data['name']}_crossk{cross_stiffness:g}_crossrest{cross_rest_scale:g}"
    )
    data["refinement"] = {
        "cross_stiffness_k": float(cross_stiffness),
        "cross_rest_scale": float(cross_rest_scale),
        "internal_stiffness_k": float(internal_stiffness),
    }
    return data


def fit_metrics(path, angles, targets, relaxation_steps, device):
    topology = load_spatial_topology(path, device)
    basis = spatial_initial_basis(topology, angles, relaxation_steps)
    base = topology["initial_stiffness"].detach().cpu().numpy()
    lower, upper = base * 10.0 ** -0.3, base * 100.0
    default = basis @ base
    default_rmse = np.sqrt(np.mean((targets - default[None, :]) ** 2, axis=1))
    fitted_rmse = []
    for target in targets:
        result = lsq_linear(basis, target, bounds=(lower, upper), max_iter=250)
        fitted_rmse.append(np.sqrt(np.mean((target - basis @ result.x) ** 2)))
    fitted_rmse = np.asarray(fitted_rmse)
    direct = np.max(np.abs(basis), axis=0) > 1e-7
    return {
        "default_mean_rmse_nm": float(np.mean(default_rmse)),
        "bounded_fit_mean_rmse_nm": float(np.mean(fitted_rmse)),
        "bounded_fit_p90_rmse_nm": float(np.percentile(fitted_rmse, 90)),
        "nonzero_direct_torque_springs": int(np.sum(direct)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--relaxation-steps", type=int, default=120)
    parser.add_argument("--profiles", type=int, default=60)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument(
        "--fast", action="store_true",
        help="Screen only the low-stiffness, low-preload region identified previously.",
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=TABLE_PATH)
    args = parser.parse_args()
    sources = sorted(args.source_dir.glob("rank_*.json"))
    if not sources:
        raise SystemExit(f"No ranked candidates found in {args.source_dir}")
    angles = np.radians(np.arange(-45.0, 46.0, 5.0))
    profiles = generate_profile_parameters(np.random.default_rng(20260813), args.profiles)
    targets = np.stack([profile_torque(angles, profile) for profile in profiles])
    rows, retained = [], []
    cross_stiffnesses = (5.0, 10.0) if args.fast else (5.0, 10.0, 20.0, 40.0)
    cross_rest_scales = (0.90, 1.00) if args.fast else (0.78, 0.90, 1.00)
    with tempfile.TemporaryDirectory(prefix="spring_refinement_") as directory:
        path = Path(directory) / "candidate.json"
        total = len(sources) * len(cross_stiffnesses) * len(cross_rest_scales)
        index = 0
        for source in sources:
            for cross_stiffness in cross_stiffnesses:
                for cross_rest_scale in cross_rest_scales:
                    index += 1
                    data = configure(source, cross_stiffness, cross_rest_scale, 90.0)
                    path.write_text(json.dumps(data), encoding="utf-8")
                    metrics = fit_metrics(
                        path, angles, targets, args.relaxation_steps, args.device
                    )
                    row = {
                        "source": source.name,
                        "cross_stiffness_k": cross_stiffness,
                        "cross_rest_scale": cross_rest_scale,
                        "internal_stiffness_k": 90.0,
                        **metrics,
                    }
                    rows.append(row)
                    retained.append((metrics["bounded_fit_mean_rmse_nm"], data, row))
                    print(
                        f"{index:3d}/{total}: {source.stem} k={cross_stiffness:g} "
                        f"rest={cross_rest_scale:.2f} default={metrics['default_mean_rmse_nm']:.2f} "
                        f"fit={metrics['bounded_fit_mean_rmse_nm']:.2f}", flush=True,
                    )
    retained.sort(key=lambda item: item[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, data, _) in enumerate(retained[:args.keep], start=1):
        (args.output_dir / f"rank_{rank:02d}.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    with args.table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for rank, (score, _, row) in enumerate(retained[:args.keep], start=1):
        print(
            f"rank={rank} fit={score:.3f} default={row['default_mean_rmse_nm']:.3f} "
            f"source={row['source']} cross_k={row['cross_stiffness_k']} "
            f"cross_rest={row['cross_rest_scale']}"
        )
    print(args.table_path)


if __name__ == "__main__":
    main()
