"""Compare the original internal fan with its preload-specific derivative."""

from pathlib import Path
import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
from topology_loader import load_network
from visualize import plot_network


def apply_preload(network, preload):
    for spring in network.springs:
        spring.rest_length = max(spring.rest_length - preload, 0.005)


def draw(ax, topology, theta, preload, title):
    network, _ = load_network(topology)
    apply_preload(network, preload)
    plot_network(network, theta, forces=None, show_forces=False, ax=ax, relax_internal=False)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral-preload-mm", type=float, default=700.0)
    parser.add_argument("--output-name", default="preload_internal_fan_comparison")
    args = parser.parse_args()
    preload_topology = PROJECT_ROOT / "topologies" / "preload" / "preload_fan_soft_015_long150.json"
    preload = args.neutral_preload_mm / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, angle_deg in zip(axes, (-30.0, 0.0, 30.0)):
        draw(
            ax,
            preload_topology,
            np.deg2rad(angle_deg),
            preload,
            f"Preload topology | {angle_deg:+.0f} deg | neutral preload {args.neutral_preload_mm:.1f} mm",
        )
    fig.suptitle("Selected 20-spring preload topology across joint angle")
    output = PROJECT_ROOT / "plots" / "preload" / "demos" / f"{args.output_name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved topology visualization: {output}")


if __name__ == "__main__":
    main()
