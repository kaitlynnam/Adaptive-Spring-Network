from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = ("time", "joint_name", "theta", "theta_dot", "tau_total")


@dataclass
class TrajectoryData:
    """Joint trajectory samples used for PEJ profile distillation."""

    time: np.ndarray
    joint_name: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray
    tau_total: np.ndarray
    terrain: np.ndarray | None = None
    policy: np.ndarray | None = None
    robot_id: np.ndarray | None = None

    @property
    def joint_names(self) -> list[str]:
        return sorted(set(str(name) for name in self.joint_name))

    def for_joint(self, joint_name: str) -> "TrajectoryData":
        mask = self.joint_name == joint_name
        if not np.any(mask):
            raise ValueError(f"joint_name {joint_name!r} was not found")
        return self._select(mask)

    def _select(self, mask: np.ndarray) -> "TrajectoryData":
        return TrajectoryData(
            time=self.time[mask],
            joint_name=self.joint_name[mask],
            theta=self.theta[mask],
            theta_dot=self.theta_dot[mask],
            tau_total=self.tau_total[mask],
            terrain=None if self.terrain is None else self.terrain[mask],
            policy=None if self.policy is None else self.policy[mask],
            robot_id=None if self.robot_id is None else self.robot_id[mask],
        )


def load_trajectory(path: str | Path) -> TrajectoryData:
    """Load trajectory samples from CSV or NPZ."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return load_trajectory_csv(path)
    if path.suffix.lower() == ".npz":
        return load_trajectory_npz(path)
    raise ValueError(f"unsupported trajectory file type: {path.suffix}")


def load_trajectory_csv(path: str | Path) -> TrajectoryData:
    """Load CSV trajectory data with required PEJ distillation columns."""

    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("trajectory CSV must include a header row")
        _require_columns(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise ValueError("trajectory CSV has no samples")

    optional = {name for name in ("terrain", "policy", "robot_id") if name in reader.fieldnames}
    return TrajectoryData(
        time=_float_column(rows, "time"),
        joint_name=np.array([row["joint_name"] for row in rows], dtype=str),
        theta=_float_column(rows, "theta"),
        theta_dot=_float_column(rows, "theta_dot"),
        tau_total=_float_column(rows, "tau_total"),
        terrain=_str_column(rows, "terrain") if "terrain" in optional else None,
        policy=_str_column(rows, "policy") if "policy" in optional else None,
        robot_id=_str_column(rows, "robot_id") if "robot_id" in optional else None,
    )


def load_trajectory_npz(path: str | Path) -> TrajectoryData:
    """Load NPZ trajectory data with arrays named after the CSV columns."""

    with np.load(path) as data:
        _require_columns(data.files)
        kwargs = {
            "time": np.asarray(data["time"], dtype=float),
            "joint_name": np.asarray(data["joint_name"], dtype=str),
            "theta": np.asarray(data["theta"], dtype=float),
            "theta_dot": np.asarray(data["theta_dot"], dtype=float),
            "tau_total": np.asarray(data["tau_total"], dtype=float),
            "terrain": np.asarray(data["terrain"], dtype=str) if "terrain" in data.files else None,
            "policy": np.asarray(data["policy"], dtype=str) if "policy" in data.files else None,
            "robot_id": np.asarray(data["robot_id"], dtype=str) if "robot_id" in data.files else None,
        }
    trajectory = TrajectoryData(**kwargs)
    _validate_lengths(trajectory)
    return trajectory


def synthetic_trajectory(
    *,
    duration: float = 20.0,
    dt: float = 0.02,
    joint_name: str = "front_thigh",
) -> TrajectoryData:
    """Generate a deterministic trajectory shaped like one distillable joint rollout."""

    time = np.arange(0.0, duration + dt, dt)
    theta = 0.35 + 0.25 * np.sin(2.0 * np.pi * 1.8 * time)
    theta_dot = np.gradient(theta, dt)
    tau_elastic = 4.0 * (theta - 0.35) + 9.0 * (theta - 0.35) ** 3
    tau_total = tau_elastic + 0.4 * np.sin(2.0 * np.pi * 3.6 * time + 0.2)
    return TrajectoryData(
        time=time,
        joint_name=np.full(time.shape, joint_name),
        theta=theta,
        theta_dot=theta_dot,
        tau_total=tau_total,
        terrain=np.full(time.shape, "synthetic"),
        policy=np.full(time.shape, "fixture"),
        robot_id=np.full(time.shape, "0"),
    )


def _require_columns(columns: list[str] | tuple[str, ...]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"trajectory data is missing required columns: {', '.join(missing)}")


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=float)


def _str_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([row[name] for row in rows], dtype=str)


def _validate_lengths(trajectory: TrajectoryData) -> None:
    n = trajectory.time.size
    arrays = (
        trajectory.joint_name,
        trajectory.theta,
        trajectory.theta_dot,
        trajectory.tau_total,
        trajectory.terrain,
        trajectory.policy,
        trajectory.robot_id,
    )
    if any(array is not None and array.size != n for array in arrays):
        raise ValueError("all trajectory arrays must have the same length")
