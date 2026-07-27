from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from physics import spring_force, torque_about_origin


ANGLE_DEGREES = np.arange(-45.0, 46.0, 5.0)


def spring_torque_basis(
    network, angles_rad, relax_internal=False, cubic_ratio=0.0,
    cubic_reference_extension=0.05,
):
    """Compute torque from each spring at 1 N/m for every angle."""
    basis = np.zeros((len(angles_rad), len(network.springs)), dtype=float)

    for angle_index, theta in enumerate(angles_rad):
        network.update_positions(theta, relax_internal=relax_internal)
        for spring_index, spring in enumerate(network.springs):
            node_a = network.nodes[spring.node_a]
            node_b = network.nodes[spring.node_b]
            force_on_a, _, stretch = spring_force(
                node_a.current_position,
                node_b.current_position,
                stiffness_k=1.0,
                rest_length=spring.rest_length,
            )
            if cubic_ratio:
                force_on_a = force_on_a * (
                    1.0 + cubic_ratio * (stretch / cubic_reference_extension) ** 2
                )

            torque = 0.0
            if node_a.type == "limb2":
                torque += torque_about_origin(node_a.current_position, force_on_a)
            if node_b.type == "limb2":
                torque += torque_about_origin(node_b.current_position, -force_on_a)
            basis[angle_index, spring_index] = torque

    return basis


def initial_stiffnesses(network):
    return np.asarray([spring.stiffness_k for spring in network.springs], dtype=float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def inverse_sigmoid(y):
    y = np.clip(y, 1e-6, 1.0 - 1e-6)
    return np.log(y / (1.0 - y))


def initialize_model(rng, input_dim, hidden_dim, output_dim, initial_k, min_k, max_k):
    """Initialize a small tanh MLP near the baseline spring stiffnesses."""
    scaled_k = (initial_k - min_k) / (max_k - min_k)
    return {
        "w1": rng.normal(0.0, 0.15, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "w2": rng.normal(0.0, 0.02, size=(hidden_dim, output_dim)),
        "b2": inverse_sigmoid(scaled_k),
    }


def angle_features(angles_rad):
    max_abs = np.max(np.abs(angles_rad))
    x = angles_rad / max_abs
    return np.column_stack([x, x**2, x**3])


def forward(model, features, min_k, max_k):
    # On some Windows Conda installations a one-row NumPy matmul can enter a
    # conflicting OpenMP/BLAS runtime and terminate the process. Evaluation is
    # sequential and normally supplies exactly one row, so compute those dot
    # products directly. Batched training/inference keeps the fast BLAS path.
    if features.ndim == 2 and features.shape[0] == 1:
        z1 = np.sum(features[0, :, None] * model["w1"], axis=0, keepdims=True) + model["b1"]
    else:
        z1 = features @ model["w1"] + model["b1"]
    hidden = np.tanh(z1)
    if hidden.ndim == 2 and hidden.shape[0] == 1:
        logits = np.sum(hidden[0, :, None] * model["w2"], axis=0, keepdims=True) + model["b2"]
    else:
        logits = hidden @ model["w2"] + model["b2"]
    sig = sigmoid(logits)
    stiffness = min_k + (max_k - min_k) * sig
    cache = {
        "features": features,
        "hidden": hidden,
        "sigmoid": sig,
    }
    return stiffness, cache


def save_model(path, model, target_name, min_k, max_k, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        target_name=target_name,
        min_k=min_k,
        max_k=max_k,
        **metadata,
        w1=model["w1"],
        b1=model["b1"],
        w2=model["w2"],
        b2=model["b2"],
    )
