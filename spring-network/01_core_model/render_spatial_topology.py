"""Render the genuine spatial internal-fan topology as a compact joint assembly."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))

from mechanics_3d import load_spatial_topology, prescribed_positions, relax_positions


TOPOLOGY_PATH = (
    PROJECT_ROOT / "topologies" / "spatial" / "internal_fan_3d_48_spring_densest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "plots" / "spatial" / "internal_fan_3d_48_spring_rest.png"
)
TYPE_COLOR = {
    "fixed": "#343a40",
    "skin1": "#2457a6",
    "skin2": "#188b67",
    "internal": "#f4a261",
    "limb1": "#31688e",
    "limb2": "#20a486",
}


def coil_points(start, stop, radius=0.012, turns=9, samples=90):
    direction = stop - start
    length = np.linalg.norm(direction)
    unit = direction / max(length, 1e-12)
    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(unit, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    side = np.cross(unit, trial)
    side /= np.linalg.norm(side)
    normal = np.cross(unit, side)
    t = np.linspace(0.0, 1.0, samples)
    taper = np.sin(np.pi * t)
    phase = 2.0 * np.pi * turns * t
    return (
        start[None, :]
        + t[:, None] * direction[None, :]
        + radius * taper[:, None]
        * (np.cos(phase)[:, None] * side + np.sin(phase)[:, None] * normal)
    )


def tapered_limb(axis, start_x, stop_x, color, angle=0.0):
    """Draw a solid rectangular link that narrows toward x=0."""
    start_scale = abs(start_x) / max(abs(start_x), abs(stop_x), 1e-9)
    stop_scale = abs(stop_x) / max(abs(start_x), abs(stop_x), 1e-9)

    def section(x, scale):
        half_y = 0.035 + 0.030 * scale
        half_z = 0.055 + 0.045 * scale
        return [
            np.array([x, -half_y, -half_z]),
            np.array([x, half_y, -half_z]),
            np.array([x, half_y, half_z]),
            np.array([x, -half_y, half_z]),
        ]

    first, second = section(start_x, start_scale), section(stop_x, stop_scale)
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        for section_points in (first, second):
            for point in section_points:
                x, z = point[0], point[2]
                point[0] = c * x + s * z
                point[2] = -s * x + c * z
    faces = [first, second]
    for index in range(4):
        following = (index + 1) % 4
        faces.append([first[index], first[following], second[following], second[index]])
    body = Poly3DCollection(
        faces, facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.34
    )
    axis.add_collection3d(body)


def skin_cylinder(axis, start_x, stop_x, radius, color, angle=0.0):
    """Draw a thin cylindrical shell rigidly attached to one limb."""
    x_grid, phi_grid = np.meshgrid(
        np.linspace(start_x, stop_x, 24),
        np.linspace(0.0, 2.0 * np.pi, 48),
    )
    y_grid = radius * np.cos(phi_grid)
    z_grid = radius * np.sin(phi_grid)
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        x_grid, z_grid = c * x_grid + s * z_grid, -s * x_grid + c * z_grid
    axis.plot_surface(
        x_grid, y_grid, z_grid, color=color, edgecolor=color,
        linewidth=0.12, alpha=0.075, shade=False,
    )


def render(topology_path=TOPOLOGY_PATH, output_path=OUTPUT_PATH, angle_degrees=0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topology = load_spatial_topology(topology_path, device)
    angle = np.radians(angle_degrees)
    theta = torch.tensor([angle], dtype=torch.float32, device=device)
    stiffness = topology["initial_stiffness"].unsqueeze(0)
    prescribed = prescribed_positions(topology, theta)
    positions = (
        relax_positions(topology, prescribed, stiffness, steps=300)[0]
        .detach().cpu().numpy()
    )
    stiffness_values = topology["initial_stiffness"].detach().cpu().numpy()
    normalization = Normalize(stiffness_values.min(), stiffness_values.max())
    spring_map = colormaps["viridis"]

    figure = plt.figure(figsize=(13, 9))
    axis = figure.add_subplot(111, projection="3d")

    split_skin = any(kind in ("skin1", "skin2") for kind in topology["node_types"])
    if split_skin:
        skin_radius = topology["data"].get("skin_radius", 0.28)
        skin_cylinder(axis, -1.05, -0.15, skin_radius, TYPE_COLOR["skin1"])
        skin_cylinder(axis, 0.15, 1.12, skin_radius, TYPE_COLOR["skin2"], angle=angle)
        # Flexible joint boot closes the wedge between the rotating rigid shells.
        boot_radius = topology["data"].get("joint_boot_radius", skin_radius)
        boot_u, boot_v = np.meshgrid(
            np.linspace(0.0, 2.0 * np.pi, 32),
            np.linspace(0.0, np.pi, 18),
        )
        axis.plot_wireframe(
            boot_radius * np.sin(boot_v) * np.cos(boot_u),
            boot_radius * np.sin(boot_v) * np.sin(boot_u),
            boot_radius * np.cos(boot_v),
            color="#7c8796", linewidth=0.28, alpha=0.09,
        )
    else:
        # Lateral mounting banks run beside the limbs instead of forming a cage.
        for y in (-0.48, 0.48):
            axis.plot(
                [-1.08, 1.08], [y, y], [0.0, 0.0],
                color="#8d99ae", linewidth=7, alpha=0.32,
            )
        fixed_indices = [
            i for i, value in enumerate(topology["node_types"]) if value == "fixed"
        ]
        for index in fixed_indices:
            point = positions[index]
            axis.plot(
                [point[0], point[0]], [point[1], point[1]], [0.0, point[2]],
                color="#adb5bd", linewidth=3, alpha=0.75,
            )

    # Solid tapered links stop on opposite sides of the shared bearing.
    tapered_limb(axis, -0.98, -0.065, TYPE_COLOR["limb1"])
    tapered_limb(axis, 0.065, 1.11, TYPE_COLOR["limb2"], angle=angle)

    # Draw exact spring paths over the translucent bodies for readability.
    for index, (a, b) in enumerate(
        zip(topology["spring_a"].tolist(), topology["spring_b"].tolist())
    ):
        coil = coil_points(positions[a], positions[b], radius=0.015)
        axis.plot(
            coil[:, 0], coil[:, 1], coil[:, 2],
            color="white", linewidth=3.2, alpha=0.72,
        )
        axis.plot(
            coil[:, 0], coil[:, 1], coil[:, 2],
            color=spring_map(normalization(stiffness_values[index])),
            linewidth=1.65, alpha=1.0,
        )

    for kind in TYPE_COLOR:
        indices = [i for i, value in enumerate(topology["node_types"]) if value == kind]
        if not indices:
            continue
        points = positions[indices]
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=70 if kind in ("fixed", "skin1", "skin2") else 90,
            color=TYPE_COLOR[kind],
            marker="s" if kind in ("fixed", "skin1", "skin2") else "o",
            edgecolor="white", linewidth=0.8, depthshade=True,
            label=f"{kind} nodes ({len(indices)})",
        )
    # Shaded barrel and end caps make the shared revolute bearing genuinely 3D.
    bearing_phi = np.linspace(0.0, 2.0 * np.pi, 64)
    bearing_half_length = topology["bearing_half_length"]
    bearing_y = np.linspace(-bearing_half_length, bearing_half_length, 18)
    bearing_phi_grid, bearing_y_grid = np.meshgrid(bearing_phi, bearing_y)
    bearing_radius = topology["bearing_radius"]
    axis.plot_surface(
        bearing_radius * np.cos(bearing_phi_grid),
        bearing_y_grid,
        bearing_radius * np.sin(bearing_phi_grid),
        color="#b9bdc2", edgecolor="none", alpha=0.96, shade=True,
    )
    for y_value in (-bearing_half_length, bearing_half_length):
        cap = [
            np.array(
                [
                    bearing_radius * np.cos(angle),
                    y_value,
                    bearing_radius * np.sin(angle),
                ]
            )
            for angle in bearing_phi
        ]
        axis.add_collection3d(
            Poly3DCollection(
                [cap], facecolor="#dfe2e5", edgecolor="#4e5052",
                linewidth=1.6, alpha=0.98,
            )
        )
    axis.plot(
        [], [], [], color="#b9bdc2", linewidth=9,
        label="shared revolute bearing",
    )

    spring_count = len(topology["spring_a"])
    assembly = (
        f"Split-Skin {spring_count}-Spring Joint"
        if split_skin else "Optimized Spatial Joint"
    )
    axis.set_title(f"{assembly} — Bend {angle_degrees:+.0f}°", pad=15)
    axis.set_xlabel("x [m]")
    # Keep the original geometry orientation; only rename the displayed axes
    # so x is lateral, y is vertical, and z is depth.
    axis.set_ylabel("z [m]")
    axis.set_zlabel("y [m]")
    axis.set_xlim(-1.12, 1.18)
    display_radius = (
        topology["data"].get("joint_boot_radius", topology["data"].get("skin_radius", 0.46))
        if split_skin else 0.68
    )
    axis.set_ylim(-1.08 * display_radius, 1.08 * display_radius)
    axis.set_zlim(-1.02 * display_radius, 1.02 * display_radius)
    axis.set_box_aspect((2.30, 2.16 * display_radius, 2.04 * display_radius))
    axis.view_init(elev=27, azim=-58, vertical_axis="z")
    axis.grid(False)
    axis.legend(loc="upper left", framealpha=0.92)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=spring_map),
        ax=axis, shrink=0.58, pad=0.07,
    )
    colorbar.set_label("baseline stiffness [N/m]")
    figure.text(
        0.5, 0.025,
        (
            "The translucent skin is split at the joint; each anchor moves rigidly with its own limb."
            if split_skin else
            "Both limbs share the central bearing; fixed anchors sit in lateral banks beside the limb paths."
        ),
        ha="center", color="0.25",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(render())
    for bend in (-45.0, 45.0):
        suffix = "negative" if bend < 0 else "positive"
        bent_output = (
            PROJECT_ROOT
            / "plots"
            / "spatial"
            / f"internal_fan_3d_30_spring_bent_{suffix}_45deg.png"
        )
        print(render(output_path=bent_output, angle_degrees=bend))
