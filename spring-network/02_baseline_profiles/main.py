from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from topology_loader import DEFAULT_TOPOLOGY_PATH, load_network


def run_demo(topology_path=DEFAULT_TOPOLOGY_PATH, relax_internal=True):
    network, topology = load_network(topology_path)
    angles = np.radians([-35.0, 0.0, 35.0])

    output_dir = PROJECT_ROOT / "plots" / "demos"
    output_dir.mkdir(exist_ok=True)

    print(f"Loaded topology: {topology['name']}")
    print(f"Nodes: {len(network.nodes)} | Springs: {len(network.springs)}")
    print(f"Internal-node relaxation: {'enabled' if relax_internal else 'disabled'}")
    print()

    fig, axes = plt.subplots(1, len(angles), figsize=(16, 5.5), constrained_layout=True)
    for ax, theta in zip(axes, angles):
        forces, spring_results, torque = network.evaluate(theta, relax_internal=relax_internal)
        print(f"theta = {np.degrees(theta):6.1f} deg | limb-2 spring torque = {torque: .4f} N*m")
        for result in spring_results:
            spring = result["spring"]
            print(
                f"  {spring.node_a:18s} -> {spring.node_b:18s} "
                f"length={result['current_length']:.3f} "
                f"stretch={result['stretch']:.3f}"
            )
        print()
        network.plot(theta, forces=forces, show_forces=True, ax=ax, relax_internal=relax_internal)

    # Saving the figure makes the first version easy to inspect in any environment.
    figure_path = output_dir / "spring_network_demo.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved plot to {figure_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 2D spring-network joint simulator.")
    parser.add_argument(
        "--topology",
        default=DEFAULT_TOPOLOGY_PATH,
        help="Path to a topology JSON file, for example spring-network/topologies/internal_fan_model.json.",
    )
    parser.add_argument(
        "--no-relax-internal",
        action="store_true",
        help="Plot the unrelaxed geometry instead of relaxed internal-node positions.",
    )
    args = parser.parse_args()
    run_demo(args.topology, relax_internal=not args.no_relax_internal)
