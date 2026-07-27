from pathlib import Path
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "02_baseline_profiles"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import forward
from energy_accounting import (
    DEFAULT_MOTORING_EFFICIENCY,
    DEFAULT_REGEN_EFFICIENCY,
    numpy_power_accounting,
    validate_efficiencies,
)
from profile_generator import (
    TERRAIN_FAMILIES,
    default_profile_named,
    generate_classified_profile_parameters,
    profile_descriptor,
    profile_torque,
)
from train_adaptive_dataset import causal_derivative, generate_motion_trajectory
from topology_loader import load_network


DEFAULT_BATCH_COUNT = 300
DEFAULT_BATCH_PROFILES_PER_FAMILY = 100
DEFAULT_BATCH_SEED = 19
DEFAULT_NETWORK_PRESET = "fan"
NETWORK_PRESETS = {
    "baseline": {
        "topology": PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "baseline_model.json",
        "adaptive_model": PROJECT_ROOT / "models" / "legacy" / "adaptive_trained_baseline_model.npz",
    },
    "fan": {
        "topology": PROJECT_ROOT / "topologies" / "adaptive_stiffness" / "internal_fan_20_spring_model.json",
        "adaptive_model": PROJECT_ROOT / "models" / "adaptive_stiffness" / "adaptive_stiffness_optimal.npz",
    },
}


def resolve_network_preset(args):
    """Resolve a named network to a compatible topology/model pair."""
    preset = NETWORK_PRESETS[args.network]
    automatic_model = args.adaptive_model is None and not args.baseline
    if args.topology is None:
        args.topology = preset["topology"]

    if args.baseline:
        args.adaptive_model = None
    elif args.adaptive_model is None:
        args.adaptive_model = preset["adaptive_model"]

    if automatic_model:
        model_path = Path(args.adaptive_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"The default torque-history model {model_path} does not exist. "
                f"Train it with --network {args.network} first."
            )
        with np.load(model_path, allow_pickle=False) as data:
            feature_type = str(data["feature_type"]) if "feature_type" in data else ""
        supported = {
            "motion_window",
            "motion_torque_window",
            "causal_motion_torque_window",
            "profile_motion_torque_window",
        }
        if feature_type not in supported:
            raise ValueError(
                f"The default {args.network} model uses unsupported feature type {feature_type!r}. "
                f"Retrain it with train_adaptive_dataset.py --network {args.network}."
            )
    return args


def integrate_trapezoid(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return np.trapz(y, x)


def synthetic_trajectory(duration, samples, amplitude_deg, frequency_hz):
    t = np.linspace(0.0, duration, samples)
    theta = np.deg2rad(amplitude_deg) * np.sin(2.0 * np.pi * frequency_hz * t)
    theta_dot = causal_derivative(theta, t)
    theta_ddot = causal_derivative(theta_dot, t)
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


def generated_target(profile, theta, theta_dot):
    del theta_dot
    params = default_profile_named(profile)
    return profile_torque(theta, params)


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
        theta_dot = causal_derivative(theta, t)
    else:
        theta_dot = np.asarray(theta_dot, dtype=float)

    theta_ddot = trajectory.get("theta_ddot")
    if theta_ddot is None or np.any(~np.isfinite(theta_ddot)):
        theta_ddot = causal_derivative(theta_dot, t)
    else:
        theta_ddot = np.asarray(theta_ddot, dtype=float)

    profile = trajectory["profile"] or args.profile
    tau_target = trajectory["tau_target"]
    if tau_target is None or np.any(~np.isfinite(tau_target)):
        tau_target = generated_target(profile, theta, theta_dot)
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
        if "torque_scale" in data:
            metadata["torque_scale"] = float(data["torque_scale"])
        if "stiffness_update_mode" in data:
            metadata["stiffness_update_mode"] = str(data["stiffness_update_mode"])
        if "duration" in data:
            metadata["duration"] = float(data["duration"])
        for key in ["profile_angle_scale", "profile_torque_scale"]:
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


def adaptive_spring_torque_over_time(
    network,
    theta,
    theta_dot,
    theta_ddot,
    tau_target,
    model_path,
    profile_params=None,
):
    model, metadata = load_adaptive_model(model_path)
    input_dim = model["w1"].shape[0]
    feature_type = metadata.get("feature_type")

    torque_history_feature_types = {
        "motion_torque_window",
        "causal_motion_torque_window",
        "profile_motion_torque_window",
    }
    if feature_type not in {"motion_window", *torque_history_feature_types}:
        raise ValueError(
            f"Unsupported adaptive feature type {feature_type!r}. "
            "Only causal motion-window models are accepted."
        )
    motion_features = motion_window_features(theta, theta_dot, theta_ddot, metadata)
    if feature_type == "profile_motion_torque_window":
        if profile_params is None:
            raise ValueError(
                "This adaptive model requires the five target-profile knots. "
                "Use a named piecewise profile or provide profile metadata with a custom trajectory."
            )
        descriptor = profile_descriptor(profile_params, metadata.get("torque_scale", 1.0))
        motion_features = np.hstack(
            (motion_features, np.repeat(descriptor[None, :], len(motion_features), axis=0))
        )
    if model["w2"].shape[1] != len(network.springs):
        raise ValueError(
            f"Adaptive model emits {model['w2'].shape[1]} stiffnesses, but topology has "
            f"{len(network.springs)} springs. Choose a matched --network preset or model."
        )
    original_stiffness = np.asarray([spring.stiffness_k for spring in network.springs], dtype=float)
    torques = []
    try:
        if feature_type in torque_history_feature_types:
            window_size = int(metadata["window_size"])
            torque_scale = max(metadata.get("torque_scale", np.max(np.abs(tau_target))), 1e-9)
            torque_history = np.zeros((window_size, 3), dtype=float)
        elif motion_features.shape[1] != input_dim:
            raise ValueError(
                f"motion_window model expects {input_dim} inputs, but generated {motion_features.shape[1]}."
            )

        previous_cycle_index = None
        for index, theta_value in enumerate(theta):
            update_mode = metadata.get("stiffness_update_mode", "timestep")
            should_update = True
            if update_mode == "period":
                if profile_params is None or "frequency_hz" not in profile_params:
                    raise ValueError("Period-limited models require profile frequency_hz metadata.")
                sample_time = index * metadata.get("duration", 1.0) / max(len(theta) - 1, 1)
                cycle_index = int(np.floor(sample_time * profile_params["frequency_hz"] + 1e-9))
                should_update = index == 0 or cycle_index != previous_cycle_index
                previous_cycle_index = cycle_index

            if should_update and feature_type in torque_history_feature_types:
                features = np.concatenate((motion_features[index], torque_history.reshape(-1)))[None, :]
                if features.shape[1] != input_dim:
                    raise ValueError(
                        f"{feature_type} model expects {input_dim} inputs, but generated {features.shape[1]}."
                    )
                stiffness_row = forward(
                    model, features, metadata["min_k"], metadata["max_k"]
                )[0][0]
            elif should_update:
                stiffness_row = forward(
                    model, motion_features[index : index + 1], metadata["min_k"], metadata["max_k"]
                )[0][0]
            for spring, stiffness_value in zip(network.springs, stiffness_row):
                spring.stiffness_k = float(stiffness_value)
            _, _, torque = network.evaluate(float(theta_value), relax_internal=True)
            torques.append(torque)
            if feature_type in torque_history_feature_types:
                realized = np.asarray(
                    [tau_target[index], torque, tau_target[index] - torque], dtype=float
                ) / torque_scale
                torque_history = np.vstack((torque_history[1:], realized))
    finally:
        for spring, stiffness_value in zip(network.springs, original_stiffness):
            spring.stiffness_k = float(stiffness_value)

    tau_spring = np.asarray(torques, dtype=float)
    return tau_spring, metadata


def evaluate_energy(
    t,
    tau_target,
    tau_spring,
    theta_dot,
    motoring_efficiency=DEFAULT_MOTORING_EFFICIENCY,
    regen_efficiency=DEFAULT_REGEN_EFFICIENCY,
):
    validate_efficiencies(motoring_efficiency, regen_efficiency)
    residual_torque = tau_target - tau_spring
    baseline = numpy_power_accounting(
        tau_target * theta_dot, motoring_efficiency, regen_efficiency
    )
    assisted = numpy_power_accounting(
        residual_torque * theta_dot, motoring_efficiency, regen_efficiency
    )

    def energy(accounting, name):
        return float(integrate_trapezoid(accounting[name], t))

    baseline_energy_burden = energy(baseline, "energy_burden_power")
    energy_burden_with_spring = energy(assisted, "energy_burden_power")
    energy_saved = baseline_energy_burden - energy_burden_with_spring
    if abs(baseline_energy_burden) < 1e-12:
        offload_fraction = 0.0
    else:
        offload_fraction = energy_saved / baseline_energy_burden

    torque_error = tau_target - tau_spring
    return {
        "residual_torque": residual_torque,
        "baseline_mechanical_power": baseline["mechanical_power"],
        "motor_mechanical_power": assisted["mechanical_power"],
        "baseline_energy_burden_power": baseline["energy_burden_power"],
        "motor_energy_burden_power": assisted["energy_burden_power"],
        "baseline_electrical_draw_power": baseline["electrical_draw_power"],
        "motor_electrical_draw_power": assisted["electrical_draw_power"],
        "baseline_braking_power": baseline["braking_mechanical_power"],
        "motor_braking_power": assisted["braking_mechanical_power"],
        "baseline_regenerated_power": baseline["regenerated_power"],
        "motor_regenerated_power": assisted["regenerated_power"],
        "baseline_energy_burden": baseline_energy_burden,
        "energy_burden_with_spring": energy_burden_with_spring,
        "baseline_net_battery_energy": energy(baseline, "net_battery_power"),
        "net_battery_energy_with_spring": energy(assisted, "net_battery_power"),
        "baseline_electrical_draw_energy": energy(baseline, "electrical_draw_power"),
        "electrical_draw_energy_with_spring": energy(assisted, "electrical_draw_power"),
        "baseline_braking_energy": energy(baseline, "braking_mechanical_power"),
        "braking_energy_with_spring": energy(assisted, "braking_mechanical_power"),
        "baseline_regenerated_energy": energy(baseline, "regenerated_power"),
        "regenerated_energy_with_spring": energy(assisted, "regenerated_power"),
        "motoring_efficiency": motoring_efficiency,
        "regen_efficiency": regen_efficiency,
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
        ("Motoring efficiency", f"{metrics['motoring_efficiency']:.3f}"),
        ("Regeneration efficiency", f"{metrics['regen_efficiency']:.3f}"),
        ("Baseline energy burden", f"{metrics['baseline_energy_burden']:.6f} J"),
        ("Energy burden with spring", f"{metrics['energy_burden_with_spring']:.6f} J"),
        ("Net battery energy with spring", f"{metrics['net_battery_energy_with_spring']:.6f} J"),
        ("Braking energy with spring", f"{metrics['braking_energy_with_spring']:.6f} J"),
        ("Regenerated energy with spring", f"{metrics['regenerated_energy_with_spring']:.6f} J"),
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
    print(f"Baseline energy burden     : {overall['baseline_energy_burden_j']:.6f} J")
    print(f"Energy burden with spring  : {overall['energy_burden_with_spring_j']:.6f} J")
    print(f"Net battery energy         : {overall['net_battery_energy_with_spring_j']:.6f} J")
    print(f"Braking energy with spring : {overall['braking_energy_with_spring_j']:.6f} J")
    print(f"Regenerated energy         : {overall['regenerated_energy_with_spring_j']:.6f} J")
    print(f"Energy saved               : {overall['energy_saved_j']:.6f} J")
    print(f"Mean absolute torque error : {overall['mean_abs_torque_error_nm']:.6f} N*m")
    print(f"Max absolute torque error  : {overall['max_abs_torque_error_nm']:.6f} N*m")
    print()

    print("Averages by profile family")
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
                "baseline_burden_j": average["baseline_energy_burden_j"],
                "assisted_burden_j": average["energy_burden_with_spring_j"],
                "braking_energy_j": average["braking_energy_with_spring_j"],
                "energy_saved_j": average["energy_saved_j"],
                "mean_abs_torque_error_nm": average["mean_abs_torque_error_nm"],
                "max_abs_torque_error_nm": average["max_abs_torque_error_nm"],
            }
        )
    print(
        _format_table(
            family_rows,
            [
                "family",
                "cases",
                "offload_pct",
                "baseline_burden_j",
                "assisted_burden_j",
                "braking_energy_j",
                "energy_saved_j",
                "mean_abs_torque_error_nm",
                "max_abs_torque_error_nm",
            ],
        )
    )


def average_metrics(rows):
    keys = [
        "baseline_energy_burden_j",
        "energy_burden_with_spring_j",
        "baseline_net_battery_energy_j",
        "net_battery_energy_with_spring_j",
        "baseline_braking_energy_j",
        "braking_energy_with_spring_j",
        "baseline_regenerated_energy_j",
        "regenerated_energy_with_spring_j",
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
        "baseline_mechanical_power",
        "motor_mechanical_power",
        "baseline_energy_burden_power",
        "motor_energy_burden_power",
        "baseline_electrical_draw_power",
        "motor_electrical_draw_power",
        "baseline_braking_power",
        "motor_braking_power",
        "baseline_regenerated_power",
        "motor_regenerated_power",
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
            metrics["baseline_mechanical_power"],
            metrics["motor_mechanical_power"],
            metrics["baseline_energy_burden_power"],
            metrics["motor_energy_burden_power"],
            metrics["baseline_electrical_draw_power"],
            metrics["motor_electrical_draw_power"],
            metrics["baseline_braking_power"],
            metrics["motor_braking_power"],
            metrics["baseline_regenerated_power"],
            metrics["motor_regenerated_power"],
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
        "motoring_efficiency",
        "regen_efficiency",
        "average_baseline_energy_burden_j",
        "average_energy_burden_with_spring_j",
        "average_baseline_net_battery_energy_j",
        "average_net_battery_energy_with_spring_j",
        "average_baseline_braking_energy_j",
        "average_braking_energy_with_spring_j",
        "average_baseline_regenerated_energy_j",
        "average_regenerated_energy_with_spring_j",
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
            "motoring_efficiency": first["motoring_efficiency"],
            "regen_efficiency": first["regen_efficiency"],
            "average_baseline_energy_burden_j": average["baseline_energy_burden_j"],
            "average_energy_burden_with_spring_j": average["energy_burden_with_spring_j"],
            "average_baseline_net_battery_energy_j": average["baseline_net_battery_energy_j"],
            "average_net_battery_energy_with_spring_j": average["net_battery_energy_with_spring_j"],
            "average_baseline_braking_energy_j": average["baseline_braking_energy_j"],
            "average_braking_energy_with_spring_j": average["braking_energy_with_spring_j"],
            "average_baseline_regenerated_energy_j": average["baseline_regenerated_energy_j"],
            "average_regenerated_energy_with_spring_j": average["regenerated_energy_with_spring_j"],
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
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), constrained_layout=True)

    axes[0].plot(t, theta, color="black")
    axes[0].set_ylabel("theta [rad]")
    axes[0].set_title(f"Trajectory evaluation: {profile}")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        t,
        tau_spring + metrics["residual_torque"],
        color="tab:green", linestyle=":", linewidth=3.0,
        label="spring + motor",
        zorder=1,
    )
    axes[1].plot(t, tau_spring, color="tab:blue", label="spring torque", linewidth=2.5, zorder=3)
    axes[1].plot(t, metrics["residual_torque"], color="tab:red", linestyle="-.", label="residual motor torque", linewidth=2.2, zorder=3)
    axes[1].plot(t, tau_target, color="black", linestyle="--", label="target torque", linewidth=2.0, zorder=4)
    axes[1].set_ylabel("torque [N*m]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    angle_order = np.argsort(theta)
    angle_deg = np.rad2deg(theta)
    edges = np.linspace(float(np.min(angle_deg)), float(np.max(angle_deg)), 31)
    bin_index = np.clip(np.digitize(angle_deg, edges) - 1, 0, len(edges) - 2)
    centers, spring_mean, motor_mean = [], [], []
    for bin_id in range(len(edges) - 1):
        mask = bin_index == bin_id
        if np.any(mask):
            centers.append(float(np.mean(angle_deg[mask])))
            spring_mean.append(float(np.mean(tau_spring[mask])))
            motor_mean.append(float(np.mean(metrics["residual_torque"][mask])))
    axes[2].scatter(angle_deg, tau_spring, s=12, color="tab:blue", alpha=0.22, label="_nolegend_", zorder=2)
    axes[2].plot(centers, spring_mean, color="tab:blue", linewidth=3.0, label="mean spring torque", zorder=3)
    axes[2].scatter(
        angle_deg,
        metrics["residual_torque"],
        s=12, color="tab:red", marker="x", alpha=0.22,
        label="_nolegend_",
    )
    axes[2].plot(centers, motor_mean, color="tab:red", linestyle="-.", linewidth=3.0, label="mean residual motor torque", zorder=3)
    axes[2].plot(angle_deg[angle_order], tau_target[angle_order], color="black", linestyle="--", linewidth=2.0, label="target torque", zorder=4)
    axes[2].set_xlabel("joint angle [deg]")
    axes[2].set_ylabel("torque [N*m]")
    axes[2].set_title("Learned torque-angle response")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t, metrics["residual_torque"], color="tab:red")
    axes[3].set_ylabel("residual [N*m]")
    axes[3].grid(True, alpha=0.25)

    axes[4].plot(t, metrics["baseline_energy_burden_power"], label="baseline burden", linewidth=1.8)
    axes[4].plot(t, metrics["motor_energy_burden_power"], label="with spring", linewidth=1.5)
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("energy burden [W]")
    axes[4].legend()
    axes[4].grid(True, alpha=0.25)

    fig.savefig(path, dpi=160)


def evaluate_case(args, profile, duration, samples, amplitude_deg, frequency_hz):
    trajectory = synthetic_trajectory(duration, samples, amplitude_deg, frequency_hz)
    tau_target = generated_target(profile, trajectory["theta"], trajectory["theta_dot"])
    params = default_profile_named(profile)
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
        profile_params=params,
    )


def evaluate_arrays(
    args,
    profile,
    t,
    theta,
    theta_dot,
    theta_ddot,
    tau_target,
    duration,
    samples,
    amplitude_deg,
    frequency_hz,
    profile_params=None,
):
    network, topology = load_network(args.topology)
    if args.adaptive_model:
        tau_spring, model_metadata = adaptive_spring_torque_over_time(
            network,
            theta,
            theta_dot,
            theta_ddot,
            tau_target,
            args.adaptive_model,
            profile_params=profile_params,
        )
        model_name = Path(args.adaptive_model).stem
    else:
        tau_spring = spring_torque_over_time(network, theta, relax_internal=not args.no_relax_internal)
        model_name = "fixed_stiffness"
    metrics = evaluate_energy(
        t,
        tau_target,
        tau_spring,
        theta_dot,
        args.motoring_efficiency,
        args.regen_efficiency,
    )
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
        "motoring_efficiency": metrics["motoring_efficiency"],
        "regen_efficiency": metrics["regen_efficiency"],
        "baseline_energy_burden_j": metrics["baseline_energy_burden"],
        "energy_burden_with_spring_j": metrics["energy_burden_with_spring"],
        "baseline_net_battery_energy_j": metrics["baseline_net_battery_energy"],
        "net_battery_energy_with_spring_j": metrics["net_battery_energy_with_spring"],
        "baseline_braking_energy_j": metrics["baseline_braking_energy"],
        "braking_energy_with_spring_j": metrics["braking_energy_with_spring"],
        "baseline_regenerated_energy_j": metrics["baseline_regenerated_energy"],
        "regenerated_energy_with_spring_j": metrics["regenerated_energy_with_spring"],
        "energy_saved_j": metrics["energy_saved"],
        "offload_pct": metrics["offload_percent"],
        "mean_abs_torque_error_nm": metrics["mean_abs_torque_error"],
        "max_abs_torque_error_nm": metrics["max_abs_torque_error"],
    }


def run_single(args):
    network, topology = load_network(args.topology)
    profile, t, theta, theta_dot, theta_ddot, tau_target = prepare_trajectory(args)
    profile_params = default_profile_named(profile) if profile and profile.startswith("piecewise_") else None
    if args.adaptive_model:
        tau_spring, model_metadata = adaptive_spring_torque_over_time(
            network,
            theta,
            theta_dot,
            theta_ddot,
            tau_target,
            args.adaptive_model,
            profile_params=profile_params,
        )
        model_name = Path(args.adaptive_model).stem
    else:
        tau_spring = spring_torque_over_time(network, theta, relax_internal=not args.no_relax_internal)
        model_name = "fixed_stiffness"
    metrics = evaluate_energy(
        t,
        tau_target,
        tau_spring,
        theta_dot,
        args.motoring_efficiency,
        args.regen_efficiency,
    )

    duration = float(t[-1] - t[0])
    topology_name = topology.get("name", Path(args.topology).stem)
    print_summary(duration, profile, topology_name, model_name, metrics)

    output_dir = Path(args.output_dir)
    suffix = f"{profile}_{Path(args.adaptive_model).stem}" if args.adaptive_model else profile
    plot_path = output_dir / f"trajectory_evaluation_{suffix}.png"
    csv_path = output_dir / f"trajectory_evaluation_{suffix}.csv"
    save_evaluation_csv(csv_path, t, theta, theta_dot, tau_target, tau_spring, metrics)
    print()
    if not args.no_plot:
        plot_evaluation(plot_path, t, theta, tau_target, tau_spring, metrics, profile)
        print(f"Saved plot to {plot_path}")
    print(f"Saved CSV to {csv_path}")


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def generate_batch_profile_parameters(args):
    rng = np.random.default_rng(args.batch_seed)
    profiles_per_family = args.profiles_per_family
    if args.batch_count is not None:
        family_count = len(TERRAIN_FAMILIES)
        if args.batch_count % family_count != 0:
            raise ValueError(
                f"--batch-count must be divisible by {family_count} so shape classes stay balanced."
            )
        profiles_per_family = args.batch_count // family_count
    return generate_classified_profile_parameters(rng, profiles_per_family)


def run_batch(args):
    profile_params = generate_batch_profile_parameters(args)

    rows = []
    for index, params in enumerate(profile_params):
        t, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params,
            duration=args.duration,
            samples=args.samples,
            seed=args.batch_seed + index,
            motion_mode=args.motion_mode,
            fixed_frequency_hz=args.fixed_frequency_hz,
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
            profile_params=params,
        )
        result["family"] = params["family"]
        if not args.no_plot and index < args.example_plots:
            example_path = Path(args.output_dir) / f"batch_example_{index + 1:02d}_{params['name']}.png"
            plot_evaluation(
                example_path, result["t"], result["theta"], result["tau_target"],
                result["tau_spring"], result["metrics"], params["name"]
            )
        rows.append(
            {
                key: result[key]
                for key in [
                    "family",
                    "model",
                    "topology",
                    "duration_s",
                    "samples",
                    "motoring_efficiency",
                    "regen_efficiency",
                    "baseline_energy_burden_j",
                    "energy_burden_with_spring_j",
                    "baseline_net_battery_energy_j",
                    "net_battery_energy_with_spring_j",
                    "baseline_braking_energy_j",
                    "braking_energy_with_spring_j",
                    "baseline_regenerated_energy_j",
                    "regenerated_energy_with_spring_j",
                    "energy_saved_j",
                    "offload_pct",
                    "mean_abs_torque_error_nm",
                    "max_abs_torque_error_nm",
                ]
            }
        )

    print_batch_summary(rows)
    summary_path = PROJECT_ROOT / "tables" / "adaptive_stiffness" / "trajectory_efficiency_summary.csv"
    save_summary_csv(summary_path, rows)
    print()
    print(f"Saved batch summary CSV to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate time-domain motor energy/offload for a spring network.")
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_PRESETS),
        default=DEFAULT_NETWORK_PRESET,
        help="Select a matched topology/adaptive-model pair (default: fan).",
    )
    parser.add_argument("--topology", default=None, help="Custom topology JSON; overrides the preset topology.")
    parser.add_argument(
        "--adaptive-model",
        default=None,
        help="Custom learned stiffness .npz model; overrides the preset adaptive model.",
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip Matplotlib output while still printing metrics and saving the evaluation CSV.",
    )
    parser.add_argument(
        "--example-plots",
        type=int,
        default=6,
        help="Number of batch trajectories exported with torque-time and torque-angle plots (default: 6).",
    )
    parser.add_argument("--trajectory", default=None, help="Optional CSV trajectory file.")
    parser.add_argument(
        "--profile",
        default="piecewise_0000",
        help="Target torque profile to use if tau_target is not supplied.",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Synthetic trajectory duration in seconds.")
    parser.add_argument(
        "--samples",
        type=int,
        default=160,
        help=(
            "Synthetic trajectory sample count (default: 160, matching the adaptive "
            "training timestep over the default five-second duration)."
        ),
    )
    parser.add_argument("--amplitude-deg", type=float, default=30.0, help="Synthetic trajectory amplitude in degrees.")
    parser.add_argument("--frequency-hz", type=float, default=1.0, help="Synthetic trajectory frequency in Hz.")
    parser.add_argument(
        "--motion-mode",
        choices=("randomized", "triangular"),
        default="randomized",
        help="Motion used for generated batch trajectories.",
    )
    parser.add_argument(
        "--fixed-frequency-hz",
        type=float,
        default=None,
        help="Use one frequency for every generated batch trajectory.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run only one trajectory using --profile, --duration, --samples, --amplitude-deg, and --frequency-hz.",
    )
    parser.add_argument(
        "--profiles-per-family",
        type=int,
        default=DEFAULT_BATCH_PROFILES_PER_FAMILY,
        help="Arbitrary profiles for each relative flat, mixed, and rough shape class (default: 100).",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=None,
        help="Optional total batch size; must be divisible by three and overrides --profiles-per-family.",
    )
    parser.add_argument("--batch-seed", type=int, default=DEFAULT_BATCH_SEED, help="Random seed for generated batch profiles.")
    parser.add_argument(
        "--motoring-efficiency",
        type=float,
        default=DEFAULT_MOTORING_EFFICIENCY,
        help="Motor/drive efficiency while delivering positive shaft power (default: 0.85).",
    )
    parser.add_argument(
        "--regen-efficiency",
        type=float,
        default=DEFAULT_REGEN_EFFICIENCY,
        help="Fraction of mechanical braking energy returned electrically (default: 0.60).",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "plots" / "adaptive_stiffness" / "trajectory_evaluation",
        help="Directory for plot and CSV outputs.",
    )
    args = parser.parse_args()
    validate_efficiencies(args.motoring_efficiency, args.regen_efficiency)
    resolve_network_preset(args)
    if args.trajectory or args.single:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
