import matplotlib.pyplot as plt
import numpy as np


NODE_COLORS = {
    "fixed": "tab:blue",
    "limb1": "tab:orange",
    "limb2": "tab:orange",
    "internal": "tab:green",
}


def plot_network(network, theta, forces=None, show_forces=True, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    network.update_positions(theta)

    # Springs are drawn first so nodes and limbs remain easy to read.
    for spring in network.springs:
        a = network.nodes[spring.node_a].current_position
        b = network.nodes[spring.node_b].current_position
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=1.2, alpha=0.75)

    limb1_end = np.array([-1.7, 0.0])
    limb2_end = np.array([1.7 * np.cos(theta), 1.7 * np.sin(theta)])
    ax.plot([0.0, limb1_end[0]], [0.0, limb1_end[1]], color="0.45", linewidth=8, solid_capstyle="round")
    ax.plot([0.0, limb2_end[0]], [0.0, limb2_end[1]], color="0.45", linewidth=8, solid_capstyle="round")

    for node_type, color in NODE_COLORS.items():
        positions = [
            node.current_position
            for node in network.nodes.values()
            if node.type == node_type
        ]
        if positions:
            positions = np.vstack(positions)
            label = "limb nodes" if node_type == "limb1" else node_type
            if node_type == "limb2":
                label = None
            ax.scatter(positions[:, 0], positions[:, 1], s=70, color=color, edgecolor="white", zorder=4, label=label)

    if forces is not None and show_forces:
        positions = []
        vectors = []
        for name, force in forces.items():
            force_norm = np.linalg.norm(force)
            if force_norm > 1e-9:
                positions.append(network.nodes[name].current_position)
                vectors.append(force)

        if positions:
            positions = np.vstack(positions)
            vectors = np.vstack(vectors)
            ax.quiver(
                positions[:, 0],
                positions[:, 1],
                vectors[:, 0],
                vectors[:, 1],
                angles="xy",
                scale_units="xy",
                scale=80.0,
                color="crimson",
                width=0.004,
                alpha=0.75,
            )

    ax.scatter([0.0], [0.0], s=180, color="black", zorder=5, label="joint")
    ax.set_title(f"Spring network at theta = {np.degrees(theta):.1f} deg")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.0, 2.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    return ax
