"""Theoretical fixed-per-profile preload ceiling with no upper travel limit."""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from scipy.optimize import lsq_linear, nnls

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from profile_generator import ANGLE_LIMIT_RAD, generate_classified_profile_parameters
from train_preload_network import build_dataset, full_relaxed_preload_torque, preload_mechanics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-per-family", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--relaxation-steps", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=121)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(args.seed)
    params = generate_classified_profile_parameters(rng, args.profiles_per_family)
    topology = PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "internal_fan_20_spring_model.json"
    angles = np.linspace(-ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD, 61)
    base_curve, sensitivity = preload_mechanics(topology, angles, 0.001)
    dataset = build_dataset(params, 5.0, args.samples, args.seed + 1000, angles, base_curve, sensitivity)

    for label, upper_m in (("10 mm", 0.010), ("50 mm", 0.050), ("unlimited", np.inf)):
        schedule = np.empty_like(dataset["sensitivity"])
        for profile_index in range(dataset["profiles"]):
            start = profile_index * dataset["samples"]
            stop = start + dataset["samples"]
            basis = np.asarray(dataset["sensitivity"][start:stop], dtype=np.float64)
            residual_target = np.asarray(
                dataset["target"][start:stop] - dataset["base"][start:stop],
                dtype=np.float64,
            )
            if np.isinf(upper_m):
                preload, _ = nnls(basis, residual_target, maxiter=2000)
            else:
                preload = lsq_linear(
                    basis, residual_target, bounds=(0.0, upper_m), max_iter=500
                ).x
            schedule[start:stop] = preload[None, :]

        local_torque = dataset["base"] + np.sum(dataset["sensitivity"] * schedule, axis=1)
        local_rmse = float(np.sqrt(np.mean((local_torque - dataset["target"]) ** 2)))
        preload_tensor = torch.as_tensor(schedule, dtype=torch.float32, device=device)
        nonlinear = full_relaxed_preload_torque(
            dataset, preload_tensor, topology, device, 4096, args.relaxation_steps
        ).cpu().numpy()
        nonlinear_rmse = float(np.sqrt(np.mean((nonlinear - dataset["target"]) ** 2)))
        print(
            f"{label:9s} | local RMSE {local_rmse:8.3f} Nm | nonlinear RMSE {nonlinear_rmse:8.3f} Nm | "
            f"mean preload {1000*np.mean(schedule):9.2f} mm | max preload {1000*np.max(schedule):9.2f} mm"
        )


if __name__ == "__main__":
    main()
