from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from python.pej.adaptive import roughness_to_q


@dataclass(frozen=True)
class StepTerrainSchedule:
    """Piecewise-constant q schedule for flat/rough/mixed terrain studies."""

    switch_time: float
    q_before: float = 0.0
    q_after: float = 1.0

    def q(self, t: np.ndarray | float) -> np.ndarray:
        t_values = np.asarray(t, dtype=float)
        return np.where(t_values < self.switch_time, self.q_before, self.q_after)


def sinusoidal_reference(amplitude: float, frequency_hz: float, offset: float = 0.0):
    """Return theta, theta_dot, and theta_ddot functions for a sinusoidal joint motion."""

    omega = 2.0 * pi * frequency_hz

    def theta(t: float) -> float:
        return float(offset + amplitude * np.sin(omega * t))

    def theta_dot(t: float) -> float:
        return float(amplitude * omega * np.cos(omega * t))

    def theta_ddot(t: float) -> float:
        return float(-amplitude * omega**2 * np.sin(omega * t))

    return theta, theta_dot, theta_ddot


def windowed_roughness_q(
    theta_dot: np.ndarray,
    *,
    window_length: int,
    min_score: float,
    max_score: float,
) -> np.ndarray:
    """Compute q from rolling joint-velocity variance."""

    if window_length <= 0:
        raise ValueError("window_length must be positive")
    theta_dot_values = np.asarray(theta_dot, dtype=float)
    scores = np.empty_like(theta_dot_values)
    for i in range(theta_dot_values.size):
        start = max(0, i + 1 - window_length)
        scores[i] = np.var(theta_dot_values[start : i + 1])
    return roughness_to_q(scores, min_score, max_score)
