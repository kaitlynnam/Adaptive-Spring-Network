from copy import deepcopy
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "02_baseline_profiles"))

from evaluate_profiles import model_torque_curve, target_profiles
from topology_loader import DEFAULT_TOPOLOGY_PATH, build_network_from_topology, load_network, save_topology


ANGLE_DEGREES = np.arange(-45.0, 46.0, 15.0)


def get_stiffnesses(topology):
    return np.asarray([spring["stiffness_k"] for spring in topology["springs"]], dtype=float)


def set_stiffnesses(topology, stiffnesses):
    for spring, stiffness in zip(topology["springs"], stiffnesses):
        spring["stiffness_k"] = float(stiffness)


def torque_curve_for_stiffnesses(base_topology, stiffnesses, angles_rad):
    topology = deepcopy(base_topology)
    set_stiffnesses(topology, stiffnesses)
    network = build_network_from_topology(topology)
    return model_torque_curve(network, angles_rad)


def loss_for_stiffnesses(base_topology, stiffnesses, angles_rad, target_torque, regularization_weight, initial_stiffnesses):
    model_torque = torque_curve_for_stiffnesses(base_topology, stiffnesses, angles_rad)
    mse = np.mean((model_torque - target_torque) ** 2)

    # Small regularization keeps inactive or weakly observed springs near their
    # starting values. It does not change the structure or add any controller.
    relative_change = (stiffnesses - initial_stiffnesses) / np.maximum(initial_stiffnesses, 1.0)
    regularization = regularization_weight * np.mean(relative_change**2)
    return float(mse + regularization), model_torque


def finite_difference_gradient(
    base_topology,
    stiffnesses,
    angles_rad,
    target_torque,
    regularization_weight,
    initial_stiffnesses,
    epsilon,
):
    gradient = np.zeros_like(stiffnesses)
    for index in range(len(stiffnesses)):
        step = np.zeros_like(stiffnesses)
        step[index] = epsilon

        loss_plus, _ = loss_for_stiffnesses(
            base_topology,
            stiffnesses + step,
            angles_rad,
            target_torque,
            regularization_weight,
            initial_stiffnesses,
        )
        loss_minus, _ = loss_for_stiffnesses(
            base_topology,
            stiffnesses - step,
            angles_rad,
            target_torque,
            regularization_weight,
            initial_stiffnesses,
        )
        gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return gradient


def optimize_stiffnesses(
    topology,
    target_name,
    iterations,
    learning_rate,
    epsilon,
    min_stiffness,
    max_stiffness,
    regularization_weight,
):
    angles_rad = np.radians(ANGLE_DEGREES)
    profiles = target_profiles(angles_rad)
    if target_name not in profiles:
        raise ValueError(f"Unknown target {target_name!r}. Options: {', '.join(profiles)}")

    target_torque = profiles[target_name]
    initial_stiffnesses = get_stiffnesses(topology)
    stiffnesses = initial_stiffnesses.copy()

    initial_loss, initial_torque = loss_for_stiffnesses(
        topology,
        stiffnesses,
        angles_rad,
        target_torque,
        regularization_weight=0.0,
        initial_stiffnesses=initial_stiffnesses,
    )
    best_loss = initial_loss
    best_stiffnesses = stiffnesses.copy()

    print(f"Target profile: {target_name}")
    print(f"Initial RMSE: {np.sqrt(initial_loss):.4f} N*m")

    for iteration in range(1, iterations + 1):
        gradient = finite_difference_gradient(
            topology,
            stiffnesses,
            angles_rad,
            target_torque,
            regularization_weight,
            initial_stiffnesses,
            epsilon,
        )
        stiffnesses = stiffnesses - learning_rate * gradient
        stiffnesses = np.clip(stiffnesses, min_stiffness, max_stiffness)

        loss, _ = loss_for_stiffnesses(
            topology,
            stiffnesses,
            angles_rad,
            target_torque,
            regularization_weight=0.0,
            initial_stiffnesses=initial_stiffnesses,
        )

        if loss < best_loss:
            best_loss = loss
            best_stiffnesses = stiffnesses.copy()

        if iteration == 1 or iteration % 25 == 0 or iteration == iterations:
            print(f"iteration {iteration:4d} | RMSE {np.sqrt(loss):8.4f} N*m")

    optimized_torque = torque_curve_for_stiffnesses(topology, best_stiffnesses, angles_rad)
    return {
        "angles_rad": angles_rad,
        "target_torque": target_torque,
        "initial_torque": initial_torque,
        "optimized_torque": optimized_torque,
        "initial_stiffnesses": initial_stiffnesses,
        "optimized_stiffnesses": best_stiffnesses,
        "initial_rmse": float(np.sqrt(initial_loss)),
        "optimized_rmse": float(np.sqrt(best_loss)),
    }


def plot_optimization(angles_deg, target_torque, initial_torque, optimized_torque, output_path):
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(angles_deg, target_torque, "k--", linewidth=2.0, label="target")
    ax.plot(angles_deg, initial_torque, "o-", linewidth=1.8, label="initial stiffness")
    ax.plot(angles_deg, optimized_torque, "s-", linewidth=1.8, label="optimized stiffness")
    ax.axhline(0.0, color="0.65", linewidth=1.0)
    ax.axvline(0.0, color="0.65", linewidth=1.0)
    ax.set_xlabel("joint angle theta [deg]")
    ax.set_ylabel("torque [N*m]")
    ax.set_title("Stiffness-only gradient descent")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)


def print_stiffness_table(topology, initial_stiffnesses, optimized_stiffnesses):
    print()
    print("Spring stiffness updates")
    print("------------------------")
    print("spring                                | initial_N/m | optimized_N/m")
    for spring, initial, optimized in zip(topology["springs"], initial_stiffnesses, optimized_stiffnesses):
        name = f"{spring['node_a']} -> {spring['node_b']}"
        print(f"{name:37s} | {initial:11.3f} | {optimized:13.3f}")


def print_torque_table(angles_deg, target_torque, initial_torque, optimized_torque):
    print()
    print("Torque curve")
    print("------------")
    print("angle_deg | target_Nm | initial_Nm | optimized_Nm")
    for angle, target, initial, optimized in zip(angles_deg, target_torque, initial_torque, optimized_torque):
        print(f"{angle:9.1f} | {target:9.3f} | {initial:10.3f} | {optimized:12.3f}")


def main():
    parser = argparse.ArgumentParser(description="Optimize spring stiffnesses with finite-difference gradient descent.")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_PATH, help="Path to the starting topology JSON file.")
    parser.add_argument("--target", default="rough_terrain", help="Target profile name from evaluate_profiles.py.")
    parser.add_argument("--iterations", type=int, default=200, help="Gradient descent iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.8, help="Gradient descent step size.")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Finite difference step in N/m.")
    parser.add_argument("--min-stiffness", type=float, default=1.0, help="Lower stiffness clamp in N/m.")
    parser.add_argument("--max-stiffness", type=float, default=250.0, help="Upper stiffness clamp in N/m.")
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=1e-3,
        help="Penalty for moving stiffnesses far from their starting values.",
    )
    args = parser.parse_args()

    _, topology = load_network(args.topology)
    result = optimize_stiffnesses(
        topology=topology,
        target_name=args.target,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        min_stiffness=args.min_stiffness,
        max_stiffness=args.max_stiffness,
        regularization_weight=args.regularization_weight,
    )

    optimized_topology = deepcopy(topology)
    optimized_topology["name"] = f"{topology['name']}_stiffness_{args.target}"
    optimized_topology["description"] = (
        f"Same spring network structure as {topology['name']}, with stiffness_k "
        f"optimized by finite-difference gradient descent for {args.target}."
    )
    set_stiffnesses(optimized_topology, result["optimized_stiffnesses"])

    output_dir = PROJECT_ROOT
    topology_path = output_dir / "topologies" / f"optimized_stiffness_{args.target}.json"
    figure_path = output_dir / "plots" / "stiffness_profiles" / f"optimized_stiffness_{args.target}.png"
    save_topology(optimized_topology, topology_path)
    plot_optimization(
        ANGLE_DEGREES,
        result["target_torque"],
        result["initial_torque"],
        result["optimized_torque"],
        figure_path,
    )

    print()
    print(f"Initial RMSE:   {result['initial_rmse']:.4f} N*m")
    print(f"Optimized RMSE: {result['optimized_rmse']:.4f} N*m")
    print_stiffness_table(topology, result["initial_stiffnesses"], result["optimized_stiffnesses"])
    print_torque_table(ANGLE_DEGREES, result["target_torque"], result["initial_torque"], result["optimized_torque"])
    print()
    print(f"Saved optimized topology to {topology_path}")
    print(f"Saved optimization plot to {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
