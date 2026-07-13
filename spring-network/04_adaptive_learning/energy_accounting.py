"""Shared bidirectional motor-energy accounting for training and evaluation."""

import numpy as np


DEFAULT_MOTORING_EFFICIENCY = 0.85
DEFAULT_REGEN_EFFICIENCY = 0.60


def validate_efficiencies(motoring_efficiency, regen_efficiency):
    if not 0.0 < motoring_efficiency <= 1.0:
        raise ValueError("motoring_efficiency must be in (0, 1].")
    if not 0.0 <= regen_efficiency <= 1.0:
        raise ValueError("regen_efficiency must be in [0, 1].")


def numpy_power_accounting(mechanical_power, motoring_efficiency, regen_efficiency):
    """Split signed shaft power into draw, recovery, and unrecovered braking.

    The primary energy-burden power is electrical draw plus braking energy that
    is not recovered. This prevents negative shaft power from disappearing from
    the offload metric while still crediting the configured regenerative share.
    """
    validate_efficiencies(motoring_efficiency, regen_efficiency)
    mechanical_power = np.asarray(mechanical_power, dtype=float)
    positive_mechanical_power = np.maximum(mechanical_power, 0.0)
    braking_mechanical_power = np.maximum(-mechanical_power, 0.0)
    electrical_draw_power = positive_mechanical_power / motoring_efficiency
    regenerated_power = braking_mechanical_power * regen_efficiency
    unrecovered_braking_power = braking_mechanical_power * (1.0 - regen_efficiency)
    return {
        "mechanical_power": mechanical_power,
        "positive_mechanical_power": positive_mechanical_power,
        "braking_mechanical_power": braking_mechanical_power,
        "electrical_draw_power": electrical_draw_power,
        "regenerated_power": regenerated_power,
        "unrecovered_braking_power": unrecovered_braking_power,
        "net_battery_power": electrical_draw_power - regenerated_power,
        "energy_burden_power": electrical_draw_power + unrecovered_braking_power,
    }


def torch_energy_burden_power(mechanical_power, motoring_efficiency, regen_efficiency):
    """Differentiable equivalent of the primary NumPy energy-burden power."""
    validate_efficiencies(motoring_efficiency, regen_efficiency)
    import torch

    positive_mechanical_power = torch.relu(mechanical_power)
    braking_mechanical_power = torch.relu(-mechanical_power)
    return (
        positive_mechanical_power / motoring_efficiency
        + braking_mechanical_power * (1.0 - regen_efficiency)
    )
