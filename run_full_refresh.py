"""Train and evaluate the canonical 60-spring passive controller."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import argparse


REPO = Path(__file__).resolve().parent
TRAINER = REPO / "spring-network" / "04_adaptive_learning" / "train_profile_conditioned_passive_3d.py"
TOPOLOGY = REPO / "spring-network" / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script in the adaptive-spring-network "
            "Conda environment."
        )

    print(f"Python: {sys.executable}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Topology: {TOPOLOGY}")

    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--topology", str(TOPOLOGY),
        "--training-profiles", "6000",
        "--test-profiles", "1200",
        "--samples", "160",
        "--motion-mode", "triangular",
        "--fixed-frequency-hz", "0.2",
        "--iterations", "10000",
        "--hidden-dim", "256",
        "--device", "cuda",
        "--relaxation-steps", "300",
        "--mechanics-batch-size", "512",
        "--mechanics-progress-batches", "10",
        "--mechanics-correction-phases", "2",
        "--mechanics-correction-profiles", "600",
        "--mechanics-correction-samples", "0",
        "--mechanics-correction-iterations", "1500",
        "--mechanics-correction-learning-rate", "0.001",
        "--nonlinear-power", "1",
        "--nonlinear-ratio", "0",
        "--progress-interval", "500",
        "--seed", str(args.seed),
        "--output-name", f"passive_skin60_seed{args.seed}",
    ]
    return subprocess.run(command, cwd=REPO, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
