"""Evaluate an existing preload-controller checkpoint on its held-out profiles."""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_preload_network as training
import profile_generator


def safe_roughness_score(profile):
    """Equivalent roughness score without NumPy's LAPACK-backed polyfit."""
    theta = np.asarray(profile["knots_theta"], dtype=float)
    torque = np.asarray(profile["knots_tau"], dtype=float)
    centered = theta - np.mean(theta)
    slope = np.sum(centered * (torque - np.mean(torque))) / max(
        float(np.sum(centered * centered)), 1e-12
    )
    line = np.mean(torque) + slope * centered
    slopes = np.diff(torque) / np.maximum(np.diff(theta), 1e-9)
    torque_range = max(float(np.ptp(torque)), 1e-9)
    linear_error = np.clip(np.sqrt(np.mean((torque - line) ** 2)) / torque_range, 0.0, 1.0)
    total_variation_ratio = np.sum(np.abs(np.diff(torque))) / torque_range
    variation_excess = np.clip((total_variation_ratio - 1.0) / 3.0, 0.0, 1.0)
    slope_rms = max(float(np.sqrt(np.mean(slopes**2))), 1e-9)
    slope_variation = np.clip(np.std(slopes) / slope_rms, 0.0, 1.0)
    signs = np.sign(slopes[np.abs(slopes) > 1e-9])
    reversal_fraction = np.count_nonzero(signs[1:] != signs[:-1]) / max(len(slopes) - 1, 1)
    return float(
        0.35 * linear_error
        + 0.30 * variation_excess
        + 0.20 * slope_variation
        + 0.15 * reversal_fraction
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--example-profiles", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.output is None:
        args.output = (
            PROJECT_ROOT
            / "plots"
            / "preload"
            / "dataset_examples"
            / f"{args.checkpoint.stem}_evaluation.png"
        )
    cfg = checkpoint["args"]
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    profile_generator.profile_roughness_score = safe_roughness_score
    rng = np.random.default_rng(cfg["seed"])
    train_params = training.generate_classified_profile_parameters(rng, cfg["profiles_per_family"])
    test_params = training.generate_classified_profile_parameters(
        rng, cfg["test_profiles_per_family"]
    )
    topology = Path(cfg["topology"])
    if not topology.exists():
        topology = PROJECT_ROOT / "topologies" / "preload" / topology.name
    angles = np.linspace(-training.ANGLE_LIMIT_RAD, training.ANGLE_LIMIT_RAD, 61)
    base, sensitivity = training.preload_mechanics(
        topology,
        angles,
        cfg["finite_difference_mm"] / 1000.0,
        reference_preload=cfg["neutral_preload_mm"] / 1000.0,
    )
    scales = training.normalization_scales(
        train_params,
        cfg["duration"],
        cfg["samples"],
        cfg["seed"] + 5000,
        cfg.get("window_size", 10),
        motion_mode=cfg.get("motion_mode", "randomized"),
        fixed_frequency_hz=cfg.get("fixed_frequency_hz"),
    )
    test = training.build_dataset(
        test_params,
        cfg["duration"],
        cfg["samples"],
        cfg["seed"] + 2000,
        angles,
        base,
        sensitivity,
        cfg.get("window_size", 10),
        scales,
        cfg.get("motion_mode", "randomized"),
        cfg.get("fixed_frequency_hz"),
    )

    network, _ = training.load_network(topology)
    nominal = np.asarray([spring.rest_length for spring in network.springs])
    minimum_rest = cfg["minimum_rest_length_mm"] / 1000.0
    limits = np.minimum(
        cfg["max_preload_mm"] / 1000.0, np.maximum(nominal - minimum_rest, 0.0)
    )
    neutral = np.minimum(cfg["neutral_preload_mm"] / 1000.0, limits)
    groups = training.preload_groups(network, cfg.get("group_mode", "four"))
    group_limits = np.asarray(
        [np.min(limits[groups == group]) for group in range(int(groups.max()) + 1)]
    )
    group_reference = np.asarray(
        [np.min(neutral[groups == group]) for group in range(int(groups.max()) + 1)]
    )
    model = training.PreloadController(
        test["features"].shape[1] + 3 * test["window_size"],
        cfg["hidden_dim"],
        groups,
        group_limits,
        group_reference,
        group_reference,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    data = training.tensors(test, device)
    with torch.no_grad():
        _, preload = training.predict(
            model,
            data,
            cfg["motoring_efficiency"],
            cfg["regen_efficiency"],
        )
    torque, lengths = training.full_relaxed_preload_torque(
        test,
        preload,
        topology,
        device,
        cfg["nonlinear_batch_size"],
        cfg["nonlinear_relaxation_steps"],
        return_lengths=True,
        tension_only=cfg.get("tension_only", False),
    )
    values = training.metrics(
        model,
        test,
        device,
        cfg["motoring_efficiency"],
        cfg["regen_efficiency"],
        torque_override=torque,
        preload_override=preload,
    )
    values.update(
        training.energy_ledger(
            test,
            torque,
            preload,
            lengths,
            topology,
            cfg["motoring_efficiency"],
            cfg["regen_efficiency"],
            tension_only=cfg.get("tension_only", False),
        )
    )
    print(
        f"offload={values['controlled_preload_offload_pct']:.4f}% | "
        f"baseline={values['baseline_motor_energy_j']:.4f} J | "
        f"residual={values['residual_motor_energy_j']:.4f} J | "
        f"motor_saved={values['motor_energy_saved_j']:.4f} J | "
        f"preload_used={values['preload_adjustment_energy_used_j']:.4f} J | "
        f"net_saved={values['net_energy_saved_after_preload_j']:.4f} J"
    )

    example_count = min(max(args.example_profiles, 0), test["profiles"])
    if example_count:
        indices = np.linspace(0, test["profiles"] - 1, example_count, dtype=int)
        fig, axes = plt.subplots(2, example_count, figsize=(4 * example_count, 7), constrained_layout=True)
        axes = np.asarray(axes).reshape(2, example_count)
        torque_np = torque.detach().cpu().numpy()
        for column, profile_index in enumerate(indices):
            start = profile_index * test["samples"]
            stop = start + test["samples"]
            time = np.linspace(0.0, test["duration"], test["samples"])
            theta_deg = np.rad2deg(test["theta"][start:stop])
            target = test["target"][start:stop]
            learned = torque_np[start:stop]
            residual = target - learned
            combined = learned + residual
            axes[0, column].plot(time, combined, color="tab:green", linestyle=":", linewidth=3.0, label="spring + motor", zorder=1)
            axes[0, column].plot(time, learned, color="tab:blue", linewidth=2.5, label="preload spring", zorder=3)
            axes[0, column].plot(time, residual, color="tab:red", linestyle="-.", linewidth=2.2, label="residual motor", zorder=3)
            axes[0, column].plot(time, target, color="black", linestyle="--", linewidth=2.0, label="target", zorder=4)
            axes[0, column].set_title(test_params[profile_index]["name"])
            axes[0, column].set_xlabel("time [s]")
            axes[0, column].set_ylabel("torque [N*m]")
            order = np.argsort(theta_deg)
            edges = np.linspace(float(np.min(theta_deg)), float(np.max(theta_deg)), 31)
            bin_index = np.clip(np.digitize(theta_deg, edges) - 1, 0, len(edges) - 2)
            centers, spring_mean, motor_mean = [], [], []
            for bin_id in range(len(edges) - 1):
                mask = bin_index == bin_id
                if np.any(mask):
                    centers.append(float(np.mean(theta_deg[mask])))
                    spring_mean.append(float(np.mean(learned[mask])))
                    motor_mean.append(float(np.mean(residual[mask])))
            axes[1, column].scatter(theta_deg, learned, s=12, color="tab:blue", alpha=0.22, label="_nolegend_", zorder=2)
            axes[1, column].plot(centers, spring_mean, color="tab:blue", linewidth=3.0, label="mean preload spring", zorder=3)
            axes[1, column].scatter(theta_deg, residual, s=12, color="tab:red", marker="x", alpha=0.22, label="_nolegend_", zorder=2)
            axes[1, column].plot(centers, motor_mean, color="tab:red", linestyle="-.", linewidth=3.0, label="mean residual motor", zorder=3)
            axes[1, column].plot(theta_deg[order], target[order], color="black", linestyle="--", linewidth=2.0, label="target", zorder=4)
            axes[1, column].set_xlabel("joint angle [deg]")
            axes[1, column].set_ylabel("torque [N*m]")
            for ax in axes[:, column]:
                ax.grid(True, alpha=0.25)
        axes[0, 0].legend()
        axes[1, 0].legend()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
        print(f"saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
