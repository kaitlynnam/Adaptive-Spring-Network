from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "02_baseline_profiles"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import angle_features, forward
from evaluate_profiles import target_profiles
from train_adaptive_dataset import generate_motion_trajectory, generate_profile_parameters
from topology_loader import DEFAULT_TOPOLOGY_PATH, load_network


DEFAULT_BATCH_COUNT = 30
DEFAULT_BATCH_SEED = 19
DEFAULT_ADAPTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "adaptive_trained_model.npz"


def synthetic_trajectory(duration, samples, amplitude_deg, frequency_hz):
    t = np.linspace(0.0, duration, samples)
    theta = np.deg2rad(amplitude_deg) * np.sin(2.0 * np.pi * frequency_hz * t)
    theta_dot = np.gradient(theta, t)
    theta_ddot = np.gradient(theta_dot, t)
    return {
        "t": t,
        "theta": theta,
        "theta_dot": theta_dot,
        "theta_ddot": theta_ddot,
        "tau_target": None,
        "profile": None,
    }


def load_trajectory_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Trajectory CSV {path} is empty.")

    required = {"t", "theta"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Trajectory CSV is missing required column(s): {', '.join(sorted(missing))}")

    def column(name):
        if name not in (reader.fieldnames or []):
            return None
        values = [row.get(name, "") for row in rows]
        if all(value == "" for value in values):
            return None
        return np.asarray([float(value) if value != "" else np.nan for value in values], dtype=float)

    profile_values = None
    if "profile" in (reader.fieldnames or []):
        profile_values = [row.get("profile", "").strip() for row in rows]
        profile_values = [value for value in profile_values if value]

    return {
        "t": column("t"),
        "theta": column("theta"),
        "theta_dot": column("theta_dot"),
        "theta_ddot": column("theta_ddot"),
        "tau_target": column("tau_target"),
        "profile": profile_values[0] if profile_values else None,
    }


def terrain_target(profile, theta, theta_dot):
    if profile == "mixed_terrain":
        return -90.0 * theta - 20.0 * theta**3 + 10.0 * np.sign(theta_dot) * theta**2

    profiles = target_profiles(theta)
    if profile not in profiles:
        raise ValueError(f"Unknown profile {profile!r}. Options: flat_terrain, rough_terrain, mixed_terrain")
    return profiles[profile]


def prepare_trajectory(args):
    if args.trajectory:
        trajectory = load_trajectory_csv(args.trajectory)
    else:
        trajectory = synthetic_trajectory(
            duration=args.duration,
            samples=args.samples,
            amplitude_deg=args.amplitude_deg,
            frequency_hz=args.frequency_hz,
        )

    t = np.asarray(trajectory["t"], dtype=float)
    theta = np.asarray(trajectory["theta"], dtype=float)
    theta_dot = trajectory["theta_dot"]
    if theta_dot is None or np.any(~np.isfinite(theta_dot)):
        theta_dot = np.gradient(theta, t)
    else:
        theta_dot = np.asarray(theta_dot, dtype=float)

    theta_ddot = trajectory.get("theta_ddot")
    if theta_ddot is None or np.any(~np.isfinite(theta_ddot)):
        theta_ddot = np.gradient(theta_dot, t)
    else:
        theta_ddot = np.asarray(theta_ddot, dtype=float)

    profile = trajectory["profile"] or args.profile
    tau_target = trajectory["tau_target"]
    if tau_target is None or np.any(~np.isfinite(tau_target)):
        tau_target = terrain_target(profile, theta, theta_dot)
    else:
        tau_target = np.asarray(tau_target, dtype=float)

    validate_trajectory(t, theta, theta_dot, theta_ddot, tau_target)
    return profile, t, theta, theta_dot, theta_ddot, tau_target


def validate_trajectory(t, theta, theta_dot, theta_ddot, tau_target):
    if not (len(t) == len(theta) == len(theta_dot) == len(theta_ddot) == len(tau_target)):
        raise ValueError("t, theta, theta_dot, theta_ddot, and tau_target must have the same length.")
    if len(t) < 2:
        raise ValueError("Trajectory must contain at least two samples.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Trajectory time values must be strictly increasing.")


def spring_torque_over_time(network, theta, relax_internal=True):
    torques = []
    for value in theta:
        _, _, torque = network.evaluate(float(value), relax_internal=relax_internal)
        torques.append(torque)
    return np.asarray(torques, dtype=float)


def load_adaptive_model(path):
    with np.load(path, allow_pickle=False) as data:
        model = {name: data[name] for name in ["w1", "b1", "w2", "b2"]}
        metadata = {
            "target_name": str(data["target_name"]),
            "min_k": float(data["min_k"]),
            "max_k": float(data["max_k"]),
        }
        if "max_abs_torque" in data:
            metadata["max_abs_torque"] = float(data["max_abs_torque"])
        if "feature_type" in data:
            metadata["feature_type"] = str(data["feature_type"])
        if "window_size" in data:
            metadata["window_size"] = int(data["window_size"])
        for key in ["theta_scale", "theta_dot_scale", "theta_ddot_scale"]:
            if key in data:
                metadata[key] = float(data[key])
    return model, metadata


def motion_window_features(theta, theta_dot, theta_ddot, metadata):
    window_size = int(metadata["window_size"])
    scaled = np.column_stack(
        [
            theta / max(metadata.get("theta_scale", np.max(np.abs(theta))), 1e-9),
            theta_dot / max(metadata.get("theta_dot_scale", np.max(np.abs(theta_dot))), 1e-9),
            theta_ddot / max(metadata.get("theta_ddot_scale", np.max(np.abs(theta_ddot))), 1e-9),
        ]
    )
    rows = []
    for index in range(len(theta)):
        start = max(0, index - window_size + 1)
        window = scaled[start : index + 1]
        if len(window) < window_size:
            pad = np.repeat(window[:1], window_size - len(window), axis=0)
            window = np.vstack([pad, window])
        rows.append(window.reshape(-1))
    return np.asarray(rows, dtype=float)


def adaptive_spring_torque_over_time(network, theta, theta_dot, theta_ddot, tau_target, model_path):
    model, metadata = load_adaptive_model(model_path)
    input_dim = model["w1"].shape[0]
    feature_type = metadata.get("feature_type")

    if feature_type == "motion_window":
        features = motion_window_features(theta, theta_dot, theta_ddot, metadata)
        if features.shape[1] != input_dim:
            raise ValueError(
                f"Motion-window model expects {input_dim} inputs, but generated {features.shape[1]}."
            )
    elif input_dim == 3:
        features = angle_features(theta)
    elif input_dim == 6:
        max_abs_theta = max(float(np.max(np.abs(theta))), 1e-9)
        theta_norm = theta / max_abs_theta
        max_abs_torque = metadata.get("max_abs_torque", max(float(np.max(np.abs(tau_target))), 1e-9))
        tau_norm = tau_target / max_abs_torque
        features = np.column_stack(
            [
                theta_norm,
                theta_norm**2,
                theta_norm**3,
                tau_norm,
                np.abs(tau_norm),
                np.sign(tau_norm),
            ]
        )
    else:
        raise ValueError(f"Unsupported adaptive model input dimension {input_dim}.")

    stiffness, _ = forward(model, features, metadata["min_k"], metadata["max_k"])
    original_stiffness = np.asarray([spring.stiffness_k for spring in network.springs], dtype=float)
    torques = []
    try:
        for theta_value, stiffness_row in zip(theta, stiffness):
            for spring, stiffness_value in zip(network.springs, stiffness_row):
                spring.stiffness_k = float(stiffness_value)
            _, _, torque = network.evaluate(float(theta_value), relax_internal=True)
            torques.append(torque)
    finally:
        for spring, stiffness_value in zip(network.springs, original_stiffness):
            spring.stiffness_k = float(stiffness_value)

    tau_spring = np.asarray(torques, dtype=float)
    return tau_spring, metadata


def evaluate_energy(t, tau_target, tau_spring, theta_dot):
    residual_torque = tau_target - tau_spring
    baseline_motor_power = np.maximum(0.0, tau_target * theta_dot)
    motor_power_with_spring = np.maximum(0.0, residual_torque * theta_dot)

    baseline_motor_energy = float(np.trapezoid(baseline_motor_power, t))
    motor_energy_with_spring = float(np.trapezoid(motor_power_with_spring, t))
    energy_saved = baseline_motor_energy - motor_energy_with_spring
    if abs(baseline_motor_energy) < 1e-12:
        offload_fraction = 0.0
    else:
        offload_fraction = energy_saved / baseline_motor_energy

    torque_error = tau_target - tau_spring
    return {
        "residual_torque": residual_torque,
        "baseline_motor_power": baseline_motor_power,
        "motor_power_with_spring": motor_power_with_spring,
        "baseline_motor_energy": baseline_motor_energy,
        "motor_energy_with_spring": motor_energy_with_spring,
        "energy_saved": energy_saved,
        "offload_fraction": offload_fraction,
        "offload_percent": 100.0 * offload_fraction,
        "mean_abs_torque_error": float(np.mean(np.abs(torque_error))),
        "max_abs_torque_error": float(np.max(np.abs(torque_error))),
    }


def print_summary(duration, profile, topology_name, model_name, metrics):
    rows = [
        ("Trajectory duration", f"{duration:.4f} s"),
        ("Profile name", profile),
        ("Topology name", topology_name),
        ("Model name", model_name),
        ("Baseline motor energy", f"{metrics['baseline_motor_energy']:.6f} J"),
        ("Motor energy with spring", f"{metrics['motor_energy_with_spring']:.6f} J"),
        ("Energy saved", f"{metrics['energy_saved']:.6f} J"),
        ("Offload percent", f"{metrics['offload_percent']:.4f} %"),
        ("Mean absolute torque error", f"{metrics['mean_abs_torque_error']:.6f} N*m"),
        ("Max absolute torque error", f"{metrics['max_abs_torque_error']:.6f} N*m"),
    ]
    width = max(len(name) for name, _ in rows)
    print("Time-domain energy/offload summary")
    print("----------------------------------")
    for name, value in rows:
        print(f"{name:<{width}} : {value}")


def print_batch_summary(rows):
    print("Batch trajectory energy/offload summary")
    print("---------------------------------------")
    print(f"Generated profiles evaluated : {len(rows)}")

    overall = average_metrics(rows)
    print()
    print("Overall averages")
    print("----------------")
    print(f"Offload percent            : {overall['offload_pct']:.4f} %")
    print(f"Baseline motor energy      : {overall['baseline_motor_energy_j']:.6f} J")
    print(f"Motor energy with spring   : {overall['motor_energy_with_spring_j']:.6f} J")
    print(f"Energy saved               : {overall['energy_saved_j']:.6f} J")
    print(f"Mean absolute torque error : {overall['mean_abs_torque_error_nm']:.6f} N*m")
    print(f"Max absolute torque error  : {overall['max_abs_torque_error_nm']:.6f} N*m")
    print()

    print("Averages by terrain family")
    print("--------------------------")
    family_rows = []
    for family in sorted({row["family"] for row in rows}):
        family_data = [row for row in rows if row["family"] == family]
        average = average_metrics(family_data)
        family_rows.append(
            {
                "family": family,
                "cases": len(family_data),
                "offload_pct": average["offload_pct"],
                "energy_saved_j": average["energy_saved_j"],
                "mean_abs_torque_error_nm": average["mean_abs_torque_error_nm"],
            }
        )
    print(_format_table(family_rows, ["family", "cases", "offload_pct", "energy_saved_j", "mean_abs_torque_error_nm"]))


def average_metrics(rows):
    keys = [
        "baseline_motor_energy_j",
        "motor_energy_with_spring_j",
        "energy_saved_j",
        "offload_pct",
        "mean_abs_torque_error_nm",
        "max_abs_torque_error_nm",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _format_table(rows, columns):
    formatted_rows = [[_format_cell(row[column]) for column in columns] for row in rows]
    widths = [len(column) for column in columns]
    for row in formatted_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    row_format = "  ".join(f"{{:<{width}}}" for width in widths)
    lines = [
        row_format.format(*columns),
        row_format.format(*["-" * width for width in widths]),
    ]
    lines.extend(row_format.format(*row) for row in formatted_rows)
    return "\n".join(lines)


def _format_cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_evaluation_csv(path, t, theta, theta_dot, tau_target, tau_spring, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "t",
        "theta",
        "theta_dot",
        "tau_target",
        "tau_spring",
        "residual_torque",
        "baseline_motor_power",
        "motor_power_with_spring",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for values in zip(
            t,
            theta,
            theta_dot,
            tau_target,
            tau_spring,
            metrics["residual_torque"],
            metrics["baseline_motor_power"],
            metrics["motor_power_with_spring"],
        ):
            writer.writerow([f"{value:.10f}" for value in values])


def save_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    columns = [
        "group",
        "model",
        "topology",
        "cases",
        "duration_s",
        "samples",
        "average_baseline_motor_energy_j",
        "average_motor_energy_with_spring_j",
        "average_energy_saved_j",
        "average_offload_pct",
        "average_mean_abs_torque_error_nm",
        "average_max_abs_torque_error_nm",
    ]

    def summary_row(group, group_rows):
        average = average_metrics(group_rows)
        return {
            "group": group,
            "model": first["model"],
            "topology": first["topology"],
            "cases": len(group_rows),
            "duration_s": first["duration_s"],
            "samples": first["samples"],
            "average_baseline_motor_energy_j": average["baseline_motor_energy_j"],
            "average_motor_energy_with_spring_j": average["motor_energy_with_spring_j"],
            "average_energy_saved_j": average["energy_saved_j"],
            "average_offload_pct": average["offload_pct"],
            "average_mean_abs_torque_error_nm": average["mean_abs_torque_error_nm"],
            "average_max_abs_torque_error_nm": average["max_abs_torque_error_nm"],
        }

    summary_rows = [summary_row("overall", rows)]
    for family in sorted({row["family"] for row in rows}):
        summary_rows.append(summary_row(family, [row for row in rows if row["family"] == family]))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: _format_cell(row[column]) for column in columns})


def plot_evaluation(path, t, theta, tau_target, tau_spring, metrics, profile):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)

    axes[0].plot(t, theta, color="black")
    axes[0].set_ylabel("theta [rad]")
    axes[0].set_title(f"Trajectory evaluation: {profile}")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, tau_target, label="target torque", linewidth=1.8)
    axes[1].plot(t, tau_spring, label="spring torque", linewidth=1.5)
    axes[1].set_ylabel("torque [N*m]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, metrics["residual_torque"], color="tab:red")
    axes[2].axhline(0.0, color="0.65", linewidth=1.0)
    axes[2].set_ylabel("residual [N*m]")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t, metrics["baseline_motor_power"], label="baseline motor power", linewidth=1.8)
    axes[3].plot(t, metrics["motor_power_with_spring"], label="with spring", linewidth=1.5)
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("positive power [W]")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.savefig(path, dpi=160)


def evaluate_case(args, profile, duration, samples, amplitude_deg, frequency_hz):
    trajectory = synthetic_trajectory(duration, samples, amplitude_deg, frequency_hz)
    tau_target = terrain_target(profile, trajectory["theta"], trajectory["theta_dot"])
    return evaluate_arrays(
        args,
        profile=profile,
        t=trajectory["t"],
        theta=trajectory["theta"],
        theta_dot=trajectory["theta_dot"],
        theta_ddot=trajectory["theta_ddot"],
        tau_target=tau_target,
        duration=duration,
        samples=samples,
        amplitude_deg=amplitude_deg,
        frequency_hz=frequency_hz,
    )


def evaluate_arrays(args, profile, t, theta, theta_dot, theta_ddot, tau_target, duration, samples, amplitude_deg, frequency_hz):
    network, topology = load_network(args.topology)
    if args.adaptive_model:
        tau_spring, model_metadata = adaptive_spring_torque_over_time(
            network,
            theta,
            theta_dot,
            theta_ddot,
            tau_target,
            args.adaptive_model,
        )
        model_name = Path(args.adaptive_model).stem
    else:
        tau_spring = spring_torque_over_time(network, theta, relax_internal=not args.no_relax_internal)
        model_name = "fixed_stiffness"
    metrics = evaluate_energy(t, tau_target, tau_spring, theta_dot)
    duration = float(t[-1] - t[0])
    topology_name = topology.get("name", Path(args.topology).stem)
    return {
        "profile": profile,
        "model": model_name,
        "topology": topology_name,
        "duration_s": duration,
        "samples": samples,
        "amplitude_deg": amplitude_deg,
        "frequency_hz": frequency_hz,
        "t": t,
        "theta": theta,
        "theta_dot": theta_dot,
        "theta_ddot": theta_ddot,
        "tau_target": tau_target,
        "tau_spring": tau_spring,
        "metrics": metrics,
        "baseline_motor_energy_j": metrics["baseline_motor_energy"],
        "motor_energy_with_spring_j": metrics["motor_energy_with_spring"],
        "energy_saved_j": metrics["energy_saved"],
        "offload_pct": metrics["offload_percent"],
        "mean_abs_torque_error_nm": metrics["mean_abs_torque_error"],
        "max_abs_torque_error_nm": metrics["max_abs_torque_error"],
    }


def run_single(args):
    network, topology = load_network(args.topology)
    profile, t, theta, theta_dot, theta_ddot, tau_target = prepare_trajectory(args)
    if args.adaptive_model:
        tau_spring, model_metadata = adaptive_spring_torque_over_time(
            network,
            theta,
            theta_dot,
            theta_ddot,
            tau_target,
            args.adaptive_model,
        )
        model_name = Path(args.adaptive_model).stem
    else:
        tau_spring = spring_torque_over_time(network, theta, relax_internal=not args.no_relax_internal)
        model_name = "fixed_stiffness"
    metrics = evaluate_energy(t, tau_target, tau_spring, theta_dot)

    duration = float(t[-1] - t[0])
    topology_name = topology.get("name", Path(args.topology).stem)
    print_summary(duration, profile, topology_name, model_name, metrics)

    output_dir = Path(args.output_dir)
    suffix = f"{profile}_{Path(args.adaptive_model).stem}" if args.adaptive_model else profile
    plot_path = output_dir / f"trajectory_evaluation_{suffix}.png"
    csv_path = output_dir / f"trajectory_evaluation_{suffix}.csv"
    plot_evaluation(plot_path, t, theta, tau_target, tau_spring, metrics, profile)
    save_evaluation_csv(csv_path, t, theta, theta_dot, tau_target, tau_spring, metrics)
    print()
    print(f"Saved plot to {plot_path}")
    print(f"Saved CSV to {csv_path}")
    plt.show()


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_batch(args):
    rng = np.random.default_rng(args.batch_seed)
    profile_params = generate_profile_parameters(rng, args.batch_count)

    rows = []
    for index, params in enumerate(profile_params):
        t, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params,
            duration=args.duration,
            samples=args.samples,
            seed=args.batch_seed + index,
        )
        result = evaluate_arrays(
            args,
            profile=params["name"],
            t=t,
            theta=theta,
            theta_dot=theta_dot,
            theta_ddot=theta_ddot,
            tau_target=tau_target,
            duration=args.duration,
            samples=args.samples,
            amplitude_deg=params["amplitude_deg"],
            frequency_hz=params["frequency_hz"],
        )
        result["family"] = params["family"]
        rows.append(
            {
                key: result[key]
                for key in [
                    "family",
                    "model",
                    "topology",
                    "duration_s",
                    "samples",
                    "baseline_motor_energy_j",
                    "motor_energy_with_spring_j",
                    "energy_saved_j",
                    "offload_pct",
                    "mean_abs_torque_error_nm",
                    "max_abs_torque_error_nm",
                ]
            }
        )

    print_batch_summary(rows)
    summary_path = PROJECT_ROOT / "tables" / "trajectory_efficiency_summary.csv"
    save_summary_csv(summary_path, rows)
    print()
    print(f"Saved batch summary CSV to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate time-domain motor energy/offload for a spring network.")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_PATH, help="Path to a topology JSON file.")
    parser.add_argument(
        "--adaptive-model",
        default=DEFAULT_ADAPTIVE_MODEL_PATH,
        help="Learned adaptive stiffness .npz model. Defaults to the active adaptive trained model.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the fixed-stiffness baseline instead of the adaptive trained model.",
    )
    parser.add_argument(
        "--no-relax-internal",
        action="store_true",
        help="Disable quasi-static internal-node relaxation.",
    )
    parser.add_argument("--trajectory", default=None, help="Optional CSV trajectory file.")
    parser.add_argument(
        "--profile",
        default="rough_terrain",
        choices=["flat_terrain", "rough_terrain", "mixed_terrain"],
        help="Target torque profile to use if tau_target is not supplied.",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Synthetic trajectory duration in seconds.")
    parser.add_argument("--samples", type=int, default=300, help="Synthetic trajectory sample count.")
    parser.add_argument("--amplitude-deg", type=float, default=30.0, help="Synthetic trajectory amplitude in degrees.")
    parser.add_argument("--frequency-hz", type=float, default=1.0, help="Synthetic trajectory frequency in Hz.")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run only one trajectory using --profile, --duration, --samples, --amplitude-deg, and --frequency-hz.",
    )
    parser.add_argument("--batch-count", type=int, default=DEFAULT_BATCH_COUNT, help="Generated trajectory/profile count for batch mode.")
    parser.add_argument("--batch-seed", type=int, default=DEFAULT_BATCH_SEED, help="Random seed for generated batch profiles.")
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "plots" / "trajectory_evaluation",
        help="Directory for plot and CSV outputs.",
    )
    args = parser.parse_args()
    if args.baseline:
        args.adaptive_model = None
    if args.trajectory or args.single:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
