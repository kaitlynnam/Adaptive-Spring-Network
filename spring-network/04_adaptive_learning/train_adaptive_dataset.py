from pathlib import Path
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import (
    ANGLE_DEGREES,
    forward,
    initialize_model,
    initial_stiffnesses,
    save_model,
    spring_torque_basis,
)
from topology_loader import DEFAULT_TOPOLOGY_PATH, load_network


def generate_profile_parameters(rng, count):
    """Generate randomized terrain-profile and motion parameters."""
    profiles = []
    for index in range(count):
        family = index % 3
        if family == 0:
            profiles.append(
                {
                    "name": f"flat_{index:04d}",
                    "family": "flat_terrain",
                    "k_neg": rng.uniform(55.0, 90.0),
                    "k_pos": rng.uniform(55.0, 90.0),
                    "cubic": rng.uniform(-8.0, 15.0),
                    "amplitude_deg": rng.uniform(18.0, 35.0),
                    "frequency_hz": rng.uniform(0.55, 1.15),
                    "phase": rng.uniform(0.0, 2.0 * np.pi),
                }
            )
        elif family == 1:
            profiles.append(
                {
                    "name": f"rough_{index:04d}",
                    "family": "rough_terrain",
                    "k_neg": rng.uniform(115.0, 165.0),
                    "k_pos": rng.uniform(115.0, 165.0),
                    "cubic": rng.uniform(35.0, 95.0),
                    "amplitude_deg": rng.uniform(20.0, 38.0),
                    "frequency_hz": rng.uniform(0.65, 1.35),
                    "phase": rng.uniform(0.0, 2.0 * np.pi),
                    "bump_count": int(rng.integers(4, 9)),
                    "noise_scale": rng.uniform(0.012, 0.03),
                }
            )
        else:
            profiles.append(
                {
                    "name": f"mixed_{index:04d}",
                    "family": "mixed_terrain",
                    "k_neg": rng.uniform(80.0, 120.0),
                    "k_pos": rng.uniform(95.0, 145.0),
                    "cubic": rng.uniform(10.0, 55.0),
                    "amplitude_deg": rng.uniform(18.0, 36.0),
                    "frequency_hz": rng.uniform(0.55, 1.25),
                    "phase": rng.uniform(0.0, 2.0 * np.pi),
                    "bump_count": int(rng.integers(3, 7)),
                    "noise_scale": rng.uniform(0.006, 0.02),
                }
            )
    rng.shuffle(profiles)
    return profiles


def profile_torque(theta, theta_dot, params):
    k = np.where(theta < 0.0, params["k_neg"], params["k_pos"])
    torque = -k * theta - params["cubic"] * theta**3
    if params["family"] == "mixed_terrain":
        torque += 10.0 * np.sign(theta_dot) * theta**2
    return torque


def smooth_noise(rng, samples, scale):
    raw = rng.normal(0.0, scale, size=samples)
    kernel_size = 13
    x = np.linspace(-2.5, 2.5, kernel_size)
    kernel = np.exp(-0.5 * x**2)
    kernel /= np.sum(kernel)
    return np.convolve(raw, kernel, mode="same")


def add_irregular_bumps(rng, t, theta, count, max_height):
    bumped = theta.copy()
    duration = float(t[-1] - t[0])
    for _ in range(count):
        center = rng.uniform(t[0] + 0.1 * duration, t[-1] - 0.1 * duration)
        width = rng.uniform(0.035, 0.14)
        height = rng.uniform(-max_height, max_height)
        bumped += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    return bumped


def generate_motion_trajectory(params, duration, samples, seed):
    """Generate terrain-specific theta(t), theta_dot(t), and theta_ddot(t)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, samples)
    amp = np.deg2rad(params["amplitude_deg"])
    freq = params["frequency_hz"]
    phase = params["phase"]

    base = amp * np.sin(2.0 * np.pi * freq * t + phase)
    harmonic = 0.18 * amp * np.sin(2.0 * np.pi * 0.5 * freq * t + 0.4 * phase)

    if params["family"] == "flat_terrain":
        theta = base + 0.08 * harmonic
    elif params["family"] == "rough_terrain":
        theta = base + harmonic
        theta = add_irregular_bumps(
            rng,
            t,
            theta,
            params["bump_count"],
            max_height=0.20 * amp,
        )
        theta += smooth_noise(rng, samples, params["noise_scale"])
    else:
        theta = base + 0.10 * harmonic
        rough_theta = add_irregular_bumps(
            rng,
            t,
            base + harmonic,
            params["bump_count"],
            max_height=0.16 * amp,
        )
        rough_theta += smooth_noise(rng, samples, params["noise_scale"])
        blend = np.zeros_like(t)
        blend[t > duration * 0.45] = 1.0
        ramp = (t > duration * 0.35) & (t <= duration * 0.45)
        blend[ramp] = (t[ramp] - duration * 0.35) / (duration * 0.10)
        theta = (1.0 - blend) * theta + blend * rough_theta

    max_angle = np.deg2rad(44.0)
    theta = np.clip(theta, -max_angle, max_angle)
    theta_dot = np.gradient(theta, t)
    theta_ddot = np.gradient(theta_dot, t)
    tau_target = profile_torque(theta, theta_dot, params)
    return t, theta, theta_dot, theta_ddot, tau_target


def motion_window_features(theta, theta_dot, theta_ddot, window_size, scales):
    """Build causal windows from recent motion only, with edge padding at the start."""
    padded = np.column_stack(
        [
            theta / scales["theta"],
            theta_dot / scales["theta_dot"],
            theta_ddot / scales["theta_ddot"],
        ]
    )
    rows = []
    for index in range(len(theta)):
        start = max(0, index - window_size + 1)
        window = padded[start : index + 1]
        if len(window) < window_size:
            pad = np.repeat(window[:1], window_size - len(window), axis=0)
            window = np.vstack([pad, window])
        rows.append(window.reshape(-1))
    return np.asarray(rows, dtype=float)


def interpolate_basis(basis_by_angle, angles_rad, theta):
    basis = np.empty((len(theta), basis_by_angle.shape[1]), dtype=float)
    for spring_index in range(basis_by_angle.shape[1]):
        basis[:, spring_index] = np.interp(
            theta,
            angles_rad,
            basis_by_angle[:, spring_index],
            left=basis_by_angle[0, spring_index],
            right=basis_by_angle[-1, spring_index],
        )
    return basis


def normalization_scales(profile_params, duration, samples, seed, window_size):
    theta_values = []
    theta_dot_values = []
    theta_ddot_values = []
    for profile_index, params in enumerate(profile_params):
        _, theta, theta_dot, theta_ddot, _ = generate_motion_trajectory(
            params,
            duration,
            samples,
            seed + profile_index,
        )
        theta_values.append(theta)
        theta_dot_values.append(theta_dot)
        theta_ddot_values.append(theta_ddot)

    def robust_scale(values, fallback):
        joined = np.concatenate(values)
        scale = float(np.percentile(np.abs(joined), 95))
        return max(scale, fallback)

    return {
        "theta": robust_scale(theta_values, np.deg2rad(1.0)),
        "theta_dot": robust_scale(theta_dot_values, 0.1),
        "theta_ddot": robust_scale(theta_ddot_values, 0.5),
        "window_size": int(window_size),
    }


def build_dataset(profile_params, angles_rad, basis_by_angle, duration, samples, window_size, scales, seed):
    rows = []
    targets = []
    basis_rows = []
    profile_indices = []
    t_rows = []
    theta_rows = []
    theta_dot_rows = []
    theta_ddot_rows = []

    for profile_index, params in enumerate(profile_params):
        t, theta, theta_dot, theta_ddot, tau_target = generate_motion_trajectory(
            params,
            duration,
            samples,
            seed + profile_index,
        )
        rows.append(motion_window_features(theta, theta_dot, theta_ddot, window_size, scales))
        targets.append(tau_target)
        basis_rows.append(interpolate_basis(basis_by_angle, angles_rad, theta))
        profile_indices.append(np.full(samples, profile_index, dtype=int))
        t_rows.append(t)
        theta_rows.append(theta)
        theta_dot_rows.append(theta_dot)
        theta_ddot_rows.append(theta_ddot)

    return {
        "features": np.vstack(rows),
        "target": np.concatenate(targets),
        "basis": np.vstack(basis_rows),
        "profile_indices": np.concatenate(profile_indices),
        "t": np.concatenate(t_rows),
        "theta": np.concatenate(theta_rows),
        "theta_dot": np.concatenate(theta_dot_rows),
        "theta_ddot": np.concatenate(theta_ddot_rows),
        "samples_per_profile": int(samples),
    }


def train_model(
    dataset,
    initial_k,
    hidden_dim,
    iterations,
    learning_rate,
    min_k,
    max_k,
    stiffness_weight,
    seed,
):
    rng = np.random.default_rng(seed)
    model = initialize_model(
        rng,
        dataset["features"].shape[1],
        hidden_dim,
        dataset["basis"].shape[1],
        initial_k,
        min_k,
        max_k,
    )

    basis = dataset["basis"]
    target = dataset["target"]
    best_model = {name: value.copy() for name, value in model.items()}
    best_loss = float("inf")
    adam_m = {name: np.zeros_like(value) for name, value in model.items()}
    adam_v = {name: np.zeros_like(value) for name, value in model.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for iteration in range(1, iterations + 1):
        stiffness, cache = forward(model, dataset["features"], min_k, max_k)
        predicted = np.sum(basis * stiffness, axis=1)
        error = predicted - target
        mse = np.mean(error**2)

        stiffness_delta = (stiffness - initial_k) / np.maximum(initial_k, 1.0)
        stiffness_penalty = stiffness_weight * np.mean(stiffness_delta**2)
        loss = mse + stiffness_penalty

        if loss < best_loss:
            best_loss = loss
            best_model = {name: value.copy() for name, value in model.items()}

        d_pred = 2.0 * error / len(error)
        d_stiffness = basis * d_pred[:, None]
        d_stiffness += stiffness_weight * 2.0 * stiffness_delta / (
            stiffness.size * np.maximum(initial_k, 1.0)
        )
        d_logits = d_stiffness * (max_k - min_k) * cache["sigmoid"] * (1.0 - cache["sigmoid"])

        d_w2 = cache["hidden"].T @ d_logits
        d_b2 = np.sum(d_logits, axis=0)
        d_hidden = d_logits @ model["w2"].T
        d_z1 = d_hidden * (1.0 - cache["hidden"] ** 2)
        d_w1 = cache["features"].T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)

        gradients = {
            "w1": d_w1,
            "b1": d_b1,
            "w2": d_w2,
            "b2": d_b2,
        }
        for name, gradient in gradients.items():
            adam_m[name] = beta1 * adam_m[name] + (1.0 - beta1) * gradient
            adam_v[name] = beta2 * adam_v[name] + (1.0 - beta2) * gradient**2
            m_hat = adam_m[name] / (1.0 - beta1**iteration)
            v_hat = adam_v[name] / (1.0 - beta2**iteration)
            model[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if iteration == 1 or iteration % 500 == 0 or iteration == iterations:
            print(f"iteration {iteration:5d} | train RMSE {np.sqrt(mse):8.4f} N*m | loss {loss:9.4f}")

    return best_model


def predict_dataset(model, dataset, min_k, max_k):
    stiffness, _ = forward(model, dataset["features"], min_k, max_k)
    predicted = np.sum(dataset["basis"] * stiffness, axis=1)
    return predicted, stiffness


def stiffness_schedule_from_model(model, dataset, min_k, max_k):
    stiffness, _ = forward(model, dataset["features"], min_k, max_k)
    return stiffness


def relaxed_torque_from_stiffness(network, theta, stiffness_schedule):
    original_stiffness = np.asarray([spring.stiffness_k for spring in network.springs], dtype=float)
    torques = []
    try:
        for theta_value, stiffness_row in zip(theta, stiffness_schedule):
            for spring, stiffness_value in zip(network.springs, stiffness_row):
                spring.stiffness_k = float(stiffness_value)
            _, _, torque = network.evaluate(float(theta_value), relax_internal=True)
            torques.append(torque)
    finally:
        for spring, stiffness_value in zip(network.springs, original_stiffness):
            spring.stiffness_k = float(stiffness_value)
    return np.asarray(torques, dtype=float)


def predict_dataset_relaxed(model, dataset, topology_path, min_k, max_k):
    stiffness = stiffness_schedule_from_model(model, dataset, min_k, max_k)
    predicted = np.empty(len(dataset["theta"]), dtype=float)
    samples = dataset["samples_per_profile"]
    profile_count = len(dataset["theta"]) // samples
    for profile_index in range(profile_count):
        start = profile_index * samples
        stop = start + samples
        network, _ = load_network(topology_path)
        predicted[start:stop] = relaxed_torque_from_stiffness(
            network,
            dataset["theta"][start:stop],
            stiffness[start:stop],
        )
    return predicted, stiffness


def fixed_stiffness_relaxed_torque(dataset, topology_path, stiffness):
    predicted = np.empty(len(dataset["theta"]), dtype=float)
    samples = dataset["samples_per_profile"]
    profile_count = len(dataset["theta"]) // samples
    for profile_index in range(profile_count):
        start = profile_index * samples
        stop = start + samples
        network, _ = load_network(topology_path)
        schedule = np.tile(stiffness, (stop - start, 1))
        predicted[start:stop] = relaxed_torque_from_stiffness(
            network,
            dataset["theta"][start:stop],
            schedule,
        )
    return predicted


def energy_offload(t, theta_dot, target, predicted):
    residual = target - predicted
    baseline_power = np.maximum(0.0, target * theta_dot)
    assisted_power = np.maximum(0.0, residual * theta_dot)
    baseline_energy = float(np.trapezoid(baseline_power, t))
    assisted_energy = float(np.trapezoid(assisted_power, t))
    if abs(baseline_energy) < 1e-12:
        return 0.0
    return 100.0 * (baseline_energy - assisted_energy) / baseline_energy


def summarize_profiles(profile_params, dataset, predicted):
    rows = []
    samples = dataset["samples_per_profile"]
    for profile_index, params in enumerate(profile_params):
        start = profile_index * samples
        stop = start + samples
        target = dataset["target"][start:stop]
        pred = predicted[start:stop]
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        rows.append(
            {
                "profile": params["name"],
                "family": params["family"],
                "rmse_nm": rmse,
                "offload_pct": energy_offload(
                    dataset["t"][start:stop],
                    dataset["theta_dot"][start:stop],
                    target,
                    pred,
                ),
                "mean_abs_residual_nm": float(np.mean(np.abs(target - pred))),
                "peak_abs_residual_nm": float(np.max(np.abs(target - pred))),
            }
        )
    return rows


def print_summary(title, rows):
    rmse = np.asarray([row["rmse_nm"] for row in rows])
    offload = np.asarray([row["offload_pct"] for row in rows])
    print()
    print(title)
    print("-" * len(title))
    print(f"profiles:        {len(rows)}")
    print(f"mean RMSE:       {np.mean(rmse):.4f} N*m")
    print(f"median RMSE:     {np.median(rmse):.4f} N*m")
    print(f"max RMSE:        {np.max(rmse):.4f} N*m")
    print(f"mean offload:    {np.mean(offload):.2f} %")
    print(f"median offload:  {np.median(offload):.2f} %")


def print_worst_cases(rows, count=8):
    print()
    print("Worst held-out trajectories")
    print("---------------------------")
    print("profile                | family              | rmse_Nm | offload_pct | peak_abs_residual_Nm")
    for row in sorted(rows, key=lambda item: item["rmse_nm"], reverse=True)[:count]:
        print(
            f"{row['profile']:22s} | {row['family']:19s} | {row['rmse_nm']:7.3f} | "
            f"{row['offload_pct']:11.3f} | {row['peak_abs_residual_nm']:20.3f}"
        )


def write_profile_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["profile", "family", "rmse_nm", "offload_pct", "mean_abs_residual_nm", "peak_abs_residual_nm"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_torque_trace_rows(path, profile_params, dataset, predicted, stiffness, network):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = dataset["samples_per_profile"]
    spring_columns = [f"k_{spring.node_a}_to_{spring.node_b}" for spring in network.springs]
    columns = [
        "profile",
        "family",
        "sample_index",
        "t",
        "theta",
        "theta_dot",
        "theta_ddot",
        "target_torque_nm",
        "spring_torque_nm",
        "residual_torque_nm",
        *spring_columns,
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for profile_index, params in enumerate(profile_params):
            start = profile_index * samples
            stop = start + samples
            for sample_index, row_index in enumerate(range(start, stop)):
                row = {
                    "profile": params["name"],
                    "family": params["family"],
                    "sample_index": sample_index,
                    "t": f"{dataset['t'][row_index]:.10f}",
                    "theta": f"{dataset['theta'][row_index]:.10f}",
                    "theta_dot": f"{dataset['theta_dot'][row_index]:.10f}",
                    "theta_ddot": f"{dataset['theta_ddot'][row_index]:.10f}",
                    "target_torque_nm": f"{dataset['target'][row_index]:.10f}",
                    "spring_torque_nm": f"{predicted[row_index]:.10f}",
                    "residual_torque_nm": f"{dataset['target'][row_index] - predicted[row_index]:.10f}",
                }
                for column, value in zip(spring_columns, stiffness[row_index]):
                    row[column] = f"{value:.10f}"
                writer.writerow(row)


def plot_test_examples(path, model, test_params, angles_rad, basis_by_angle, duration, samples, window_size, scales, min_k, max_k, seed, topology_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for profile_index, (ax, params) in enumerate(zip(axes, test_params[:6])):
        network, _ = load_network(topology_path)
        dataset = build_dataset(
            [params],
            angles_rad,
            basis_by_angle,
            duration,
            samples,
            window_size,
            scales,
            seed + profile_index,
        )
        stiffness = stiffness_schedule_from_model(model, dataset, min_k, max_k)
        predicted = relaxed_torque_from_stiffness(network, dataset["theta"], stiffness)
        ax.plot(dataset["t"], dataset["target"], "k--", linewidth=1.8, label="target")
        ax.plot(dataset["t"], predicted, linewidth=1.5, label="learned")
        ax.set_title(f"{params['family']} / {params['name']}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("torque [N*m]")
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Train motion-window adaptive spring stiffnesses on generated terrain trajectories.")
    parser.add_argument("--topology", default=DEFAULT_TOPOLOGY_PATH, help="Starting topology JSON file.")
    parser.add_argument("--train-profiles", type=int, default=90, help="Number of generated training trajectories.")
    parser.add_argument("--test-profiles", type=int, default=30, help="Number of held-out generated trajectories.")
    parser.add_argument("--duration", type=float, default=5.0, help="Trajectory duration in seconds.")
    parser.add_argument("--samples", type=int, default=160, help="Samples per generated trajectory.")
    parser.add_argument("--window-size", type=int, default=10, help="Recent motion samples used as neural-network input.")
    parser.add_argument("--iterations", type=int, default=1500, help="Gradient descent iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam step size.")
    parser.add_argument("--hidden-dim", type=int, default=40, help="Hidden units in the neural stiffness model.")
    parser.add_argument("--min-stiffness", type=float, default=1.0, help="Minimum learned stiffness in N/m.")
    parser.add_argument("--max-stiffness", type=float, default=300.0, help="Maximum learned stiffness in N/m.")
    parser.add_argument("--stiffness-weight", type=float, default=2e-4, help="Penalty for moving far from baseline stiffnesses.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    args = parser.parse_args()

    network, topology = load_network(args.topology)
    angles_rad = np.radians(ANGLE_DEGREES)
    basis_by_angle = spring_torque_basis(network, angles_rad, relax_internal=True)
    base_k = initial_stiffnesses(network)

    rng = np.random.default_rng(args.seed)
    all_params = generate_profile_parameters(rng, args.train_profiles + args.test_profiles)
    train_params = all_params[: args.train_profiles]
    test_params = all_params[args.train_profiles :]

    scales = normalization_scales(
        train_params,
        args.duration,
        args.samples,
        args.seed + 10_000,
        args.window_size,
    )
    train_dataset = build_dataset(
        train_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.seed + 20_000,
    )
    test_dataset = build_dataset(
        test_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.seed + 30_000,
    )

    baseline_fixed_train = fixed_stiffness_relaxed_torque(train_dataset, args.topology, base_k)
    baseline_fixed_test = fixed_stiffness_relaxed_torque(test_dataset, args.topology, base_k)
    train_baseline_rmse = float(np.sqrt(np.mean((baseline_fixed_train - train_dataset["target"]) ** 2)))
    test_baseline_rmse = float(np.sqrt(np.mean((baseline_fixed_test - test_dataset["target"]) ** 2)))

    print(f"Loaded topology: {topology['name']}")
    print(f"Training trajectories: {len(train_params)} | test trajectories: {len(test_params)}")
    print(f"Samples per trajectory: {args.samples} | motion window: {args.window_size} samples")
    print(f"Feature count: {train_dataset['features'].shape[1]} ({args.window_size} * theta/theta_dot/theta_ddot)")
    print("Training torque basis: relaxed internal-node geometry")
    print("Reported metrics: full relaxed network evaluation")
    print(f"Fixed-stiffness baseline train RMSE: {train_baseline_rmse:.4f} N*m")
    print(f"Fixed-stiffness baseline test RMSE:  {test_baseline_rmse:.4f} N*m")

    model = train_model(
        dataset=train_dataset,
        initial_k=base_k,
        hidden_dim=args.hidden_dim,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        min_k=args.min_stiffness,
        max_k=args.max_stiffness,
        stiffness_weight=args.stiffness_weight,
        seed=args.seed,
    )

    train_pred, train_stiffness = predict_dataset_relaxed(
        model,
        train_dataset,
        args.topology,
        args.min_stiffness,
        args.max_stiffness,
    )
    test_pred, test_stiffness = predict_dataset_relaxed(
        model,
        test_dataset,
        args.topology,
        args.min_stiffness,
        args.max_stiffness,
    )
    train_rows = summarize_profiles(train_params, train_dataset, train_pred)
    test_rows = summarize_profiles(test_params, test_dataset, test_pred)

    print_summary("Training-set performance", train_rows)
    print_summary("Held-out test performance", test_rows)
    print_worst_cases(test_rows)

    output_dir = PROJECT_ROOT
    model_path = output_dir / "models" / "adaptive_trained_model.npz"
    train_table_path = output_dir / "tables" / "adaptive_trained_model_train_results.csv"
    test_table_path = output_dir / "tables" / "adaptive_trained_model_test_results.csv"
    torque_trace_path = output_dir / "tables" / "adaptive_trained_model_test_torque_trace.csv"
    figure_path = output_dir / "plots" / "dataset_examples" / "adaptive_trained_model_test_examples.png"

    save_model(
        model_path,
        model,
        "adaptive_trained_model",
        args.min_stiffness,
        args.max_stiffness,
        feature_type="motion_window",
        window_size=args.window_size,
        theta_scale=scales["theta"],
        theta_dot_scale=scales["theta_dot"],
        theta_ddot_scale=scales["theta_ddot"],
        duration=args.duration,
        samples=args.samples,
    )
    write_profile_rows(train_table_path, train_rows)
    write_profile_rows(test_table_path, test_rows)
    write_torque_trace_rows(torque_trace_path, test_params, test_dataset, test_pred, test_stiffness, network)
    plot_test_examples(
        figure_path,
        model,
        test_params,
        angles_rad,
        basis_by_angle,
        args.duration,
        args.samples,
        args.window_size,
        scales,
        args.min_stiffness,
        args.max_stiffness,
        args.seed + 40_000,
        args.topology,
    )

    print()
    print(f"Saved dataset-trained model to {model_path}")
    print(f"Saved train results to {train_table_path}")
    print(f"Saved test results to {test_table_path}")
    print(f"Saved test torque trace to {torque_trace_path}")
    print(f"Saved test example plot to {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
