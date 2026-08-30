"""Measure relaxed baseline torque while sweeping the topology rest-length scale."""

from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from mechanics_3d import load_spatial_topology, torque_and_residual  # noqa: E402


DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"
)


def measure(topology_path, rest_length_scale, relaxation_steps, angle_samples, device):
    topology = load_spatial_topology(topology_path, device)
    configured_scale = float(topology["data"].get("rest_length_scale", 1.0))
    topology["rest_lengths"] *= rest_length_scale / configured_scale
    angles = torch.linspace(
        -np.pi / 4, np.pi / 4, angle_samples,
        dtype=torch.float32, device=device,
    )
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(angle_samples, 1)
    torque, residual, _ = torque_and_residual(
        topology, angles, stiffness, relaxation_steps=relaxation_steps
    )
    return {
        "rest_length_scale": rest_length_scale,
        "nominal_preload_pct": 100.0 * (1.0 - rest_length_scale),
        "baseline_torque_rms_nm": float(torch.sqrt(torch.mean(torque.square()))),
        "baseline_peak_abs_torque_nm": float(torch.max(torch.abs(torque))),
        "mean_force_residual_n": float(torch.mean(residual)),
        "max_force_residual_n": float(torch.max(residual)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--relaxation-steps", type=int, default=160)
    parser.add_argument("--angle-samples", type=int, default=9)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        measure(
            args.topology, scale, args.relaxation_steps, args.angle_samples, args.device
        )
        for scale in args.scales
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"scale={row['rest_length_scale']:.4f} "
            f"rms={row['baseline_torque_rms_nm']:.6f} Nm "
            f"peak={row['baseline_peak_abs_torque_nm']:.6f} Nm"
        )
    print(args.output)


if __name__ == "__main__":
    main()
