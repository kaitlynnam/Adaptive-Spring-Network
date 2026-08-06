"""Export joint torque trajectories from an Isaac Lab checkpoint rollout.

Run this from the repository root with an Isaac Lab-enabled Python, for example:

    <ISAAC_LAB_ROOT>\\isaaclab.bat -p spring-network\\06_isaaclab_export\\export_pretrained_rollout.py ^
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 ^
        --checkpoint path\\to\\model.pt ^
        --num-envs 64 ^
        --steps 2000

The exported data is written under spring-network/data/isaaclab_rollouts by
default.  The CSV uses one row per (step, env, joint), while the NPZ preserves
batched arrays for faster loading.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "isaaclab_rollouts"


def _import_app_launcher():
    try:
        from isaaclab.app import AppLauncher

        return AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher

        return AppLauncher


AppLauncher = _import_app_launcher()


def _import_isaaclab_runtime():
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

        return (
            RslRlVecEnvWrapper,
            get_checkpoint_path,
            get_published_pretrained_checkpoint,
            load_cfg_from_registry,
            parse_env_cfg,
        )
    except ImportError:
        from omni.isaac.lab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

        return (
            RslRlVecEnvWrapper,
            get_checkpoint_path,
            get_published_pretrained_checkpoint,
            load_cfg_from_registry,
            parse_env_cfg,
        )


def _import_rsl_rl_runner():
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise ImportError(
            "Could not import rsl_rl.runners.OnPolicyRunner. Run this script from the same "
            "Isaac Lab environment used for RSL-RL training/playback."
        ) from exc
    return OnPolicyRunner


def prepare_rsl_rl_compatibility(agent_cfg: Any, checkpoint: Path) -> tuple[Any, Path]:
    """Convert deprecated Isaac Lab configs/checkpoints for the installed RSL-RL."""
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
    except ImportError:
        return agent_cfg, checkpoint

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    checkpoint = Path(handle_deprecated_rsl_rl_checkpoint(str(checkpoint), installed_version))
    return agent_cfg, checkpoint


@dataclass
class ExportMetadata:
    task: str
    checkpoint: str
    output_prefix: str
    num_envs: int
    steps_requested: int
    decimation: int
    torque_source: str
    policy_label: str
    device: str
    dt: float | None
    joint_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an Isaac Lab RSL-RL checkpoint and export Go2 joint torque trajectories."
    )
    parser.add_argument("--task", required=True, help="Isaac Lab task id, e.g. Isaac-Velocity-Flat-Unitree-Go2-v0.")
    parser.add_argument("--checkpoint", default=None, help="Path to an RSL-RL checkpoint .pt file.")
    parser.add_argument(
        "--use-pretrained-checkpoint",
        action="store_true",
        help="Use Isaac Lab's published pretrained RSL-RL checkpoint for --task.",
    )
    parser.add_argument("--load-run", default=None, help="RSL-RL log run name if --checkpoint is omitted.")
    parser.add_argument("--load-checkpoint", default="model_.*.pt", help="Checkpoint glob if --checkpoint is omitted.")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel envs to replay.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of policy steps to record.")
    parser.add_argument("--decimation", type=int, default=1, help="Record every Nth policy step.")
    parser.add_argument(
        "--torque-source",
        choices=("applied", "computed"),
        default="applied",
        help="Which torque tensor to use as tau_total in the CSV/NPZ export.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for exported rollout files.")
    parser.add_argument("--output-prefix", default=None, help="Base filename. Defaults to task plus checkpoint name.")
    parser.add_argument("--policy-label", default=None, help="Label stored in the export metadata.")
    parser.add_argument("--experiment-name", default=None, help="Override experiment name for checkpoint lookup.")
    parser.add_argument("--seed", type=int, default=None, help="Override env seed if supported by the task config.")
    parser.add_argument("--terrain-label", default="", help="Optional terrain/condition label stored in CSV rows.")
    parser.add_argument(
        "--terrain-mode",
        choices=("task", "plane"),
        default="task",
        help="Use the task terrain or override it with a flat plane while preserving the task observation layout.",
    )
    parser.add_argument(
        "--write-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write long-form CSV in addition to NPZ.",
    )
    parser.add_argument(
        "--write-extra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include actions, commands, base velocities, dones, and contact forces when available.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return vars(config)
    if isinstance(config, dict):
        return config
    raise TypeError(f"Unsupported RSL-RL config object: {type(config)!r}")


def set_config_attr(config: Any, name: str, value: Any) -> None:
    if value is not None and hasattr(config, name):
        setattr(config, name, value)


def resolve_checkpoint(args: argparse.Namespace, agent_cfg: Any, get_checkpoint_path, get_published_pretrained_checkpoint) -> Path:
    if args.use_pretrained_checkpoint:
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", args.task)
        if checkpoint is None:
            raise FileNotFoundError(f"No published pretrained RSL-RL checkpoint found for task {args.task!r}.")
        return Path(checkpoint).expanduser().resolve()

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    log_root_path = Path("logs") / "rsl_rl"
    experiment_name = args.experiment_name or getattr(agent_cfg, "experiment_name", args.task)
    load_run = args.load_run or getattr(agent_cfg, "load_run", ".*")
    checkpoint_pattern = args.load_checkpoint or getattr(agent_cfg, "load_checkpoint", "model_.*.pt")
    checkpoint = get_checkpoint_path(str(log_root_path / experiment_name), load_run, checkpoint_pattern)
    return Path(checkpoint).expanduser().resolve()


def safe_tensor(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return torch.as_tensor(value).detach().cpu().numpy()
    except Exception:
        return None


def get_robot(env: Any) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        raise AttributeError("Could not find env.unwrapped.scene.")
    for key in ("robot", "articulation"):
        try:
            return scene[key]
        except KeyError:
            pass
    raise KeyError("Could not find a 'robot' or 'articulation' entity in env.unwrapped.scene.")


def get_terrain_metadata(env: Any, num_envs: int) -> dict[str, np.ndarray]:
    """Return auditable per-environment Isaac Lab terrain identifiers."""
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    try:
        terrain = scene["terrain"]
    except (KeyError, TypeError):
        return {}

    type_indices = safe_tensor(getattr(terrain, "terrain_types", None))
    levels = safe_tensor(getattr(terrain, "terrain_levels", None))
    origins = safe_tensor(getattr(terrain, "env_origins", None))
    cfg = getattr(terrain, "cfg", None)
    generator_cfg = getattr(cfg, "terrain_generator", None)
    terrain_kind = str(getattr(cfg, "terrain_type", "unknown"))

    if type_indices is None:
        type_indices = np.full(num_envs, -1, dtype=np.int32)
    else:
        type_indices = np.asarray(type_indices, dtype=np.int32).reshape(-1)
    if levels is None:
        levels = np.full(num_envs, -1, dtype=np.int32)
    else:
        levels = np.asarray(levels, dtype=np.int32).reshape(-1)
    if origins is None:
        origins = np.full((num_envs, 3), np.nan, dtype=np.float32)
    else:
        origins = np.asarray(origins, dtype=np.float32).reshape(num_envs, 3)

    family_names = np.full(num_envs, terrain_kind, dtype="U64")
    column_families: list[str] = []
    if generator_cfg is not None:
        sub_terrains = getattr(generator_cfg, "sub_terrains", {})
        names = list(sub_terrains.keys())
        proportions = np.asarray(
            [float(getattr(sub_cfg, "proportion", 0.0)) for sub_cfg in sub_terrains.values()], dtype=float
        )
        num_cols = int(getattr(generator_cfg, "num_cols", 0))
        curriculum = bool(getattr(generator_cfg, "curriculum", False))
        if curriculum and names and num_cols > 0 and proportions.sum() > 0:
            cumulative = np.cumsum(proportions / proportions.sum())
            column_families = [
                names[int(np.min(np.where(column / num_cols + 0.001 < cumulative)[0]))]
                for column in range(num_cols)
            ]
            family_names = np.asarray(
                [column_families[index] if 0 <= index < len(column_families) else "unknown_generator" for index in type_indices],
                dtype="U64",
            )
        else:
            # Random terrain generation does not retain the sampled family for
            # every cell, so do not invent a label from the column index.
            family_names[:] = "unknown_generator"

    return {
        "terrain_type_index": type_indices,
        "terrain_level": levels,
        "terrain_family": family_names,
        "terrain_origin": origins,
        "terrain_column_families": np.asarray(column_families, dtype="U64"),
    }


def get_dt(env: Any) -> float | None:
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "step_dt"):
        return float(unwrapped.step_dt)
    if hasattr(unwrapped, "physics_dt"):
        return float(unwrapped.physics_dt)
    try:
        return float(unwrapped.sim.get_physics_dt())
    except Exception:
        return None


def get_joint_names(robot: Any, joint_count: int) -> list[str]:
    names = getattr(robot, "joint_names", None)
    if names:
        return [str(name) for name in names]
    data_names = getattr(getattr(robot, "data", None), "joint_names", None)
    if data_names:
        return [str(name) for name in data_names]
    return [f"joint_{index}" for index in range(joint_count)]


def get_command_array(env: Any) -> np.ndarray | None:
    unwrapped = getattr(env, "unwrapped", env)
    command_manager = getattr(unwrapped, "command_manager", None)
    if command_manager is None:
        return None
    try:
        command = command_manager.get_command("base_velocity")
    except Exception:
        return None
    return safe_tensor(command)


def collect_rollout(args: argparse.Namespace, env: Any, policy: Any, checkpoint: Path) -> tuple[dict[str, np.ndarray], ExportMetadata]:
    robot = get_robot(env)
    robot_data = robot.data
    dt = get_dt(env)

    # RSL-RL 5 policies consume the TensorDict returned by Isaac Lab's wrapper.
    # Older wrappers may return a plain tensor or an (observations, extras) pair.
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    first_joint_pos = safe_tensor(robot_data.joint_pos)
    if first_joint_pos is None or first_joint_pos.ndim != 2:
        raise ValueError("Expected robot.data.joint_pos with shape [num_envs, num_joints].")

    num_envs, joint_count = first_joint_pos.shape
    joint_names = get_joint_names(robot, joint_count)
    terrain_metadata = get_terrain_metadata(env, num_envs)
    recorded_steps = (args.steps + args.decimation - 1) // args.decimation

    arrays: dict[str, list[np.ndarray]] = {
        "time": [],
        "theta": [],
        "theta_dot": [],
        "tau_applied": [],
        "tau_computed": [],
        "tau_total": [],
    }
    extra_arrays: dict[str, list[np.ndarray]] = {}

    for step in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        if step % args.decimation != 0:
            continue

        joint_pos = safe_tensor(robot_data.joint_pos)
        joint_vel = safe_tensor(robot_data.joint_vel)
        applied = safe_tensor(getattr(robot_data, "applied_torque", None))
        computed = safe_tensor(getattr(robot_data, "computed_torque", None))
        if computed is None:
            computed = applied
        if applied is None:
            applied = computed
        if joint_pos is None or joint_vel is None or applied is None or computed is None:
            raise ValueError("Could not read joint_pos, joint_vel, applied_torque, and computed_torque from robot.data.")

        tau_total = applied if args.torque_source == "applied" else computed
        sample_time = float(step if dt is None else step * dt)

        arrays["time"].append(np.full((num_envs,), sample_time, dtype=np.float64))
        arrays["theta"].append(joint_pos.astype(np.float32, copy=False))
        arrays["theta_dot"].append(joint_vel.astype(np.float32, copy=False))
        arrays["tau_applied"].append(applied.astype(np.float32, copy=False))
        arrays["tau_computed"].append(computed.astype(np.float32, copy=False))
        arrays["tau_total"].append(tau_total.astype(np.float32, copy=False))

        if args.write_extra:
            extras = {
                "actions": safe_tensor(actions),
                "commands": get_command_array(env),
                "base_lin_vel": safe_tensor(getattr(robot_data, "root_lin_vel_b", None)),
                "base_ang_vel": safe_tensor(getattr(robot_data, "root_ang_vel_b", None)),
                "dones": safe_tensor(dones),
                "contact_forces": safe_tensor(getattr(robot_data, "net_contact_forces_w", None)),
            }
            for key, value in extras.items():
                if value is not None:
                    extra_arrays.setdefault(key, []).append(value.astype(np.float32, copy=False))

    if not arrays["theta"]:
        raise RuntimeError("No rollout samples were recorded. Check --steps and --decimation.")

    stacked = {key: np.stack(values, axis=0) for key, values in arrays.items()}
    for key, values in extra_arrays.items():
        if len(values) == len(arrays["theta"]):
            stacked[key] = np.stack(values, axis=0)

    policy_label = args.policy_label or checkpoint.stem
    metadata = ExportMetadata(
        task=args.task,
        checkpoint=str(checkpoint),
        output_prefix=args.output_prefix or "",
        num_envs=num_envs,
        steps_requested=args.steps,
        decimation=args.decimation,
        torque_source=args.torque_source,
        policy_label=policy_label,
        device=str(getattr(env.unwrapped, "device", "")),
        dt=dt,
        joint_names=joint_names,
    )
    stacked["joint_names"] = np.asarray(joint_names, dtype=str)
    stacked["robot_id"] = np.asarray([str(index) for index in range(num_envs)], dtype=str)
    stacked["policy"] = np.asarray(policy_label, dtype=str)
    stacked["terrain"] = np.asarray(args.terrain_label, dtype=str)
    stacked.update(terrain_metadata)
    return stacked, metadata


def write_npz(path: Path, arrays: dict[str, np.ndarray], metadata: ExportMetadata) -> None:
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(json.dumps(asdict(metadata), indent=2), dtype=str)
    np.savez_compressed(path, **payload)


def write_csv(path: Path, arrays: dict[str, np.ndarray], metadata: ExportMetadata, terrain_label: str) -> None:
    time = arrays["time"]
    theta = arrays["theta"]
    theta_dot = arrays["theta_dot"]
    tau_total = arrays["tau_total"]
    tau_applied = arrays["tau_applied"]
    tau_computed = arrays["tau_computed"]
    actions = arrays.get("actions")
    terrain_families = arrays.get("terrain_family")
    terrain_types = arrays.get("terrain_type_index")
    terrain_levels = arrays.get("terrain_level")
    terrain_origins = arrays.get("terrain_origin")
    joint_names = metadata.joint_names

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time",
                "step_index",
                "robot_id",
                "joint_name",
                "joint_index",
                "theta",
                "theta_dot",
                "tau_total",
                "tau_applied",
                "tau_computed",
                "action",
                "terrain",
                "terrain_family",
                "terrain_type_index",
                "terrain_level",
                "terrain_origin_x",
                "terrain_origin_y",
                "terrain_origin_z",
                "policy",
            ]
        )
        for step_index in range(theta.shape[0]):
            for env_index in range(theta.shape[1]):
                for joint_index, joint_name in enumerate(joint_names):
                    action_value = ""
                    if actions is not None and actions.ndim == 3 and joint_index < actions.shape[2]:
                        action_value = f"{float(actions[step_index, env_index, joint_index]):.9g}"
                    family = str(terrain_families[env_index]) if terrain_families is not None else terrain_label
                    type_index = int(terrain_types[env_index]) if terrain_types is not None else ""
                    level = int(terrain_levels[env_index]) if terrain_levels is not None else ""
                    origin = terrain_origins[env_index] if terrain_origins is not None else ("", "", "")
                    writer.writerow(
                        [
                            f"{float(time[step_index, env_index]):.9g}",
                            step_index,
                            env_index,
                            joint_name,
                            joint_index,
                            f"{float(theta[step_index, env_index, joint_index]):.9g}",
                            f"{float(theta_dot[step_index, env_index, joint_index]):.9g}",
                            f"{float(tau_total[step_index, env_index, joint_index]):.9g}",
                            f"{float(tau_applied[step_index, env_index, joint_index]):.9g}",
                            f"{float(tau_computed[step_index, env_index, joint_index]):.9g}",
                            action_value,
                            terrain_label,
                            family,
                            type_index,
                            level,
                            origin[0],
                            origin[1],
                            origin[2],
                            metadata.policy_label,
                        ]
                    )


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None

    try:
        import gymnasium as gym

        (
            RslRlVecEnvWrapper,
            get_checkpoint_path,
            get_published_pretrained_checkpoint,
            load_cfg_from_registry,
            parse_env_cfg,
        ) = _import_isaaclab_runtime()
        OnPolicyRunner = _import_rsl_rl_runner()

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        set_config_attr(env_cfg, "seed", args.seed)
        if args.terrain_mode == "plane":
            env_cfg.scene.terrain.terrain_type = "plane"
            env_cfg.scene.terrain.terrain_generator = None
            if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        set_config_attr(agent_cfg, "device", args.device)
        set_config_attr(agent_cfg, "seed", args.seed)

        checkpoint = resolve_checkpoint(args, agent_cfg, get_checkpoint_path, get_published_pretrained_checkpoint)
        agent_cfg, checkpoint = prepare_rsl_rl_compatibility(agent_cfg, checkpoint)
        print(f"[INFO] Checkpoint: {checkpoint}", flush=True)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env)

        print("[INFO] Creating RSL-RL policy runner...", flush=True)
        runner = OnPolicyRunner(env, config_to_dict(agent_cfg), log_dir=None, device=getattr(agent_cfg, "device", args.device))
        print("[INFO] Loading checkpoint...", flush=True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=getattr(agent_cfg, "device", args.device))

        print(f"[INFO] Collecting {args.steps} rollout steps...", flush=True)
        arrays, metadata = collect_rollout(args, env, policy, checkpoint)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.output_prefix or f"{args.task}_{checkpoint.stem}".replace("/", "_").replace("\\", "_").replace(":", "_")
        metadata.output_prefix = prefix
        npz_path = output_dir / f"{prefix}.npz"
        csv_path = output_dir / f"{prefix}.csv"
        metadata_path = output_dir / f"{prefix}_metadata.json"

        write_npz(npz_path, arrays, metadata)
        if args.write_csv:
            write_csv(csv_path, arrays, metadata, args.terrain_label)
        metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

        print(f"[INFO] Exported {arrays['theta'].shape[0]} recorded steps, {arrays['theta'].shape[1]} envs, {arrays['theta'].shape[2]} joints.")
        print(f"[INFO] NPZ:      {npz_path}")
        if args.write_csv:
            print(f"[INFO] CSV:      {csv_path}")
        print(f"[INFO] Metadata: {metadata_path}")
    except BaseException:
        # Isaac Sim shutdown can stall on Windows. Print the original failure
        # before shutdown so it is not hidden behind simulation_app.close().
        print("[ERROR] Rollout export failed before Isaac Sim shutdown:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()
        simulation_app.close()


if __name__ == "__main__":
    main()
