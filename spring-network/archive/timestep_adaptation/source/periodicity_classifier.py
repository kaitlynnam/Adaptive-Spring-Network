import numpy as np


PERIODICITY_CLASSIFICATION = "cycle_repeatability_v1"


def _normalized_cycle_error(cycles):
    """Return phase-aligned cycle variation normalized by signal range."""
    cycles = np.asarray(cycles, dtype=float)
    mean_cycle = np.mean(cycles, axis=0)
    scale = max(float(np.ptp(cycles)), float(np.std(cycles)), 1e-9)
    return float(np.sqrt(np.mean((cycles - mean_cycle[None, :]) ** 2)) / scale)


def periodicity_score(t, theta, torque, nominal_frequency_hz, phase_samples=100):
    """Score cycle-to-cycle repeatability on [0, 1]; larger is more periodic.

    Cycles are cut using the known nominal stride period and resampled onto a
    common 0--100% phase axis. Both joint angle and joint torque must repeat.
    This deliberately measures temporal repeatability rather than the geometric
    complexity of the torque-angle curve.
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    torque = np.asarray(torque, dtype=float)
    if not (t.shape == theta.shape == torque.shape):
        raise ValueError("t, theta, and torque must have matching shapes")
    if nominal_frequency_hz <= 0.0:
        raise ValueError("nominal_frequency_hz must be positive")

    period = 1.0 / float(nominal_frequency_hz)
    cycle_count = int(np.floor((t[-1] - t[0]) / period))
    if cycle_count < 2:
        raise ValueError("trajectory must contain at least two complete nominal cycles")

    phase = np.linspace(0.0, 1.0, int(phase_samples), endpoint=False)
    theta_cycles = []
    torque_cycles = []
    for cycle_index in range(cycle_count):
        sample_t = t[0] + (cycle_index + phase) * period
        theta_cycles.append(np.interp(sample_t, t, theta))
        torque_cycles.append(np.interp(sample_t, t, torque))

    theta_error = _normalized_cycle_error(theta_cycles)
    torque_error = _normalized_cycle_error(torque_cycles)
    combined_error = 0.5 * (theta_error + torque_error)
    score = 1.0 / (1.0 + combined_error)
    return {
        "periodicity_score": float(score),
        "cycle_repeatability_error": float(combined_error),
        "theta_cycle_error": float(theta_error),
        "torque_cycle_error": float(torque_error),
        "complete_cycles": int(cycle_count),
    }
