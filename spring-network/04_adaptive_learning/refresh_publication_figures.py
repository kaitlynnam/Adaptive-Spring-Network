"""Re-export saved benchmark arrays using the shared publication style.

No training, evaluation, or resampling is performed. Legacy convergence PNGs
without saved histories are preserved rather than reconstructed from pixels.
"""

from pathlib import Path
import argparse
import numpy as np

from benchmark_period_adaptive_deployment import save_benchmark_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-experiments", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tables = root / "tables" / "period_adaptive_3d"
    plots = root / "plots" / "period_adaptive_3d"
    paths = sorted(tables.glob("*_complete_data.npz")) if args.all_experiments else [
        tables / "period_adaptive_3d_60spring_bounded_extended_many_profiles_complete_data.npz"
    ]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"theta", "target_torque", "spring_torque", "rmse", "offload_pct"}
            if not required.issubset(data.files) or data["spring_torque"].ndim != 3:
                continue
            name = path.name.removesuffix("_complete_data.npz")
            save_benchmark_figures(
                plots, name, {"theta": data["theta"], "target": data["target_torque"]},
                data["spring_torque"], data["rmse"], data["offload_pct"],
            )
        print(f"Refreshed {name}", flush=True)


if __name__ == "__main__":
    main()
