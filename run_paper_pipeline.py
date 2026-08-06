"""Run all canonical seeds and regenerate the complete passive paper set."""

from pathlib import Path
import argparse
import subprocess
import sys


REPO = Path(__file__).resolve().parent


def run(*arguments: str, log=None) -> None:
    subprocess.run(
        [sys.executable, "-u", *arguments],
        cwd=REPO,
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    log = args.log.open("w", encoding="utf-8", buffering=1) if args.log else None
    seed = 101
    run(str(REPO / "run_full_refresh.py"), "--seed", str(seed), log=log)

    checkpoint = REPO / "spring-network" / "models" / "profile_conditioned_passive_3d" / "passive_skin60_seed101.npz"
    run(
        str(REPO / "spring-network" / "04_adaptive_learning" / "generate_profile_passive_3d_figures.py"),
        str(checkpoint),
        "--output-stem", "canonical_seed101",
        log=log,
    )
    run(str(REPO / "generate_passive_paper_summary.py"), log=log)
    if log is not None:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
