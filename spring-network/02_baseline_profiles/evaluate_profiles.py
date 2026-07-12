from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from profile_generator import default_piecewise_profiles
from topology_loader import DEFAULT_TOPOLOGY_PATH, load_network


def integrate_trapezoid(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return np.trapz(y, x)


def target_profiles(theta):
    """Random piecewise-linear torque profiles used for comparison.

    Each target is defined by random torque-angle knot points and evaluated by
    linear interpolation between those knots.
    """
    return default_piecewise_profiles(theta)


def model_torque_curve(network, angles_rad):
    torques = []
    for theta in angles_rad:
        _, _, torque = network.evaluate(theta)
        torques.append(torque)
    return np.asarray(torques)


def print_spring_stiffnesses(network):
    print("Spring stiffnesses")
    print("------------------")
    for index, spring in enumerate(network.springs, start=1):
        print(
            f"{index:2d}. {spring.node_a:18s} -> {spring.node_b:18s} "
            f"k = {spring.stiffness_k:7.2f} N/m"
        )
    print()


def profile_error(model_torque, target_torque):
    error = model_torque - target_torque
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def load_offload_metrics(angles_rad, model_torque, target_torque):
    residual_torque = target_torque - model_torque
    baseline_effort = float(integrate_trapezoid(np.abs(target_torque), angles_rad))
    residual_effort = float(integrate_trapezoid(np.abs(residual_torque), angles_rad))
    net_saved = baseline_effort - residual_effort
    offload_pct = 0.0 if baseline_effort == 0.0 else net_saved / baseline_effort * 100.0
    return {
        "baseline_effort": baseline_effort,
        "residual_effort": residual_effort,
        "net_saved": net_saved,
        "offload_pct": offload_pct,
    }


def plot_profiles(angles_deg, model_torque, profiles, output_path, model_name):
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(angles_deg, model_torque, "ko-", linewidth=2.5, label=model_name)

    for name, target in profiles.items():
        ax.plot(angles_deg, target, "--", linewidth=1.6, label=name)

    ax.axhline(0.0, color="0.65", linewidth=1.0)
    ax.axvline(0.0, color="0.65", linewidth=1.0)
    ax.set_xlabel("joint angle theta [deg]")
    ax.set_ylabel("torque [N*m]")
    ax.set_title(f"{model_name} vs target torque profiles")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)


def run(topology_path=DEFAULT_TOPOLOGY_PATH):
    network, topology = load_network(topology_path)
    angles_deg = np.arange(-45.0, 46.0, 15.0)
    angles_rad = np.radians(angles_deg)
    model_name = topology.get("name", Path(topology_path).stem)
    model_torque = model_torque_curve(network, angles_rad)
    profiles = target_profiles(angles_rad)

    output_dir = PROJECT_ROOT / "plots" / "torque_profiles"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "torque_profile_comparison_baseline_model.png"

    print(f"Loaded topology: {topology['name']}")
    print(f"Nodes: {len(network.nodes)} | Springs: {len(network.springs)}")
    print()
    print_spring_stiffnesses(network)

    print("Model torque curve")
    print("------------------")
    print("angle_deg | model_torque_Nm")
    for angle, torque in zip(angles_deg, model_torque):
        print(f"{angle:9.1f} | {torque:15.3f}")
    print()

    print("Profile match errors")
    print("--------------------")
    print("profile         | rmse_Nm | mae_Nm | max_abs_Nm")
    for name, target in profiles.items():
        metrics = profile_error(model_torque, target)
        print(
            f"{name:15s} | {metrics['rmse']:7.3f} | "
            f"{metrics['mae']:6.3f} | {metrics['max_abs']:10.3f}"
        )
    print()

    print("Quasi-static offload")
    print("--------------------")
    print("profile         | baseline_effort | residual_effort | offload_pct")
    offload_values = []
    for name, target in profiles.items():
        metrics = load_offload_metrics(angles_rad, model_torque, target)
        offload_values.append(metrics["offload_pct"])
        print(
            f"{name:15s} | {metrics['baseline_effort']:15.4f} | "
            f"{metrics['residual_effort']:15.4f} | {metrics['offload_pct']:10.2f}"
        )
    average_offload = float(np.mean(offload_values)) if offload_values else 0.0
    print(f"Average offload across profiles: {average_offload:.2f}%")
    print()

    plot_profiles(angles_deg, model_torque, profiles, output_path, model_name)
    print(f"Saved profile comparison plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the current spring network to target torque profiles.")
    parser.add_argument(
        "--topology",
        default=DEFAULT_TOPOLOGY_PATH,
        help="Path to a topology JSON file.",
    )
    args = parser.parse_args()
    run(args.topology)
