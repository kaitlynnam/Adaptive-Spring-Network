"""Create a standalone interactive HTML deployment simulation from torque-time data."""

from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from adaptive_model import ANGLE_DEGREES
from deploy_period_adaptive_3d import deploy, load_checkpoint
from mechanics_3d import load_spatial_topology, prescribed_positions, relax_positions
from period_adaptive_support import interpolate_basis, spatial_initial_basis
from profile_generator import generate_profile_parameters
from train_period_adaptive_3d import DEFAULT_TOPOLOGY, build_period_dataset

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "period_adaptive_3d" / "period_adaptive_3d_60spring_bounded_extended.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "plots" / "period_adaptive_3d" / "interactive_simulation.html"


def dataset_from_csv(path, metadata, angles, basis):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("input CSV is empty")
    columns = set(rows[0])
    time_name = next((x for x in ("time_s", "time", "t") if x in columns), None)
    torque_name = next((x for x in ("target_torque_nm", "target_torque", "torque_nm", "torque") if x in columns), None)
    angle_name = next((x for x in ("angle_deg", "joint_angle_deg", "theta_deg") if x in columns), None)
    if time_name is None or torque_name is None:
        raise ValueError("CSV requires time_s and target_torque_nm columns")
    time = np.asarray([float(row[time_name]) for row in rows])
    target = np.asarray([float(row[torque_name]) for row in rows])
    if np.any(np.diff(time) <= 0):
        raise ValueError("CSV time values must be strictly increasing")
    period_seconds = float(metadata["period_seconds"])
    samples = int(metadata["samples_per_period"])
    duration = time[-1] - time[0]
    periods = int(np.floor(duration / period_seconds + 1e-8))
    if periods < 1:
        raise ValueError("CSV must contain at least one complete configured period")
    supplied_angle = None if angle_name is None else np.radians(
        np.asarray([float(row[angle_name]) for row in rows])
    )
    output = {key: [] for key in ("t", "theta", "theta_dot", "theta_ddot", "target", "basis")}
    for period in range(periods):
        start = time[0] + period * period_seconds
        absolute_t = np.linspace(start, start + period_seconds, samples)
        local_t = absolute_t - start
        if supplied_angle is None:
            phase = np.mod(local_t / period_seconds, 1.0)
            theta = np.deg2rad(45.0) * (1.0 - 4.0 * np.abs(phase - 0.5))
        else:
            theta = np.interp(absolute_t, time, supplied_angle)
        theta_dot = np.gradient(theta, local_t)
        theta_ddot = np.gradient(theta_dot, local_t)
        output["t"].append(local_t)
        output["theta"].append(theta)
        output["theta_dot"].append(theta_dot)
        output["theta_ddot"].append(theta_ddot)
        output["target"].append(np.interp(absolute_t, time, target))
        output["basis"].append(interpolate_basis(basis, angles, theta))
    result = {key: np.asarray(value, dtype=float) for key, value in output.items()}
    result.update(samples_per_period=samples, period_seconds=period_seconds,
                  torque_scale=float(metadata["torque_scale"]))
    return result


def demo_dataset(metadata, angles, basis, periods, seed, profile_count=1):
    """Generate consecutive profile blocks, each repeated for ``periods`` periods."""
    unique_profiles = generate_profile_parameters(
        np.random.default_rng(seed), profile_count
    )
    profiles = [
        dict(profile)
        for profile in unique_profiles
        for _ in range(periods)
    ]
    dataset = build_period_dataset(
        profiles, angles, basis, float(metadata["period_seconds"]),
        int(metadata["samples_per_period"]), seed + 10_000,
        motion_mode=str(metadata["motion_mode"]),
        frequency_hz=1.0 / float(metadata["period_seconds"]),
        torque_scale=float(metadata["torque_scale"]),
    )
    dataset["profile_index"] = np.repeat(np.arange(profile_count), periods)
    dataset["period_in_profile"] = np.tile(np.arange(periods), profile_count)
    dataset["profile_count"] = profile_count
    dataset["periods_per_profile"] = periods
    return dataset


def spring_xyz(positions, spring_a, spring_b):
    x, y, z = [], [], []
    for a, b in zip(spring_a, spring_b):
        x.extend((positions[a, 0], positions[b, 0], None))
        y.extend((positions[a, 1], positions[b, 1], None))
        z.extend((positions[a, 2], positions[b, 2], None))
    return x, y, z


def stiffness_colors(values, color_min, color_max):
    log_min = np.log10(max(color_min, 1e-6))
    log_max = np.log10(max(color_max, color_min + 1e-6))
    fractions = np.clip(
        (np.log10(np.maximum(values, 1e-6)) - log_min) / max(log_max - log_min, 1e-9),
        0.0, 1.0,
    )
    return sample_colorscale("Turbo", fractions.tolist())


def rotate_points(points, axis, angle):
    """Rotate physical-coordinate points about an axis through the origin."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    points = np.asarray(points, dtype=float)
    return (points * np.cos(angle)
            + np.cross(axis, points) * np.sin(angle)
            + np.outer(points @ axis, axis) * (1.0 - np.cos(angle)))


def display_xyz(points):
    """Keep geometry coordinates unchanged; viewer labels Y and Z oppositely."""
    points = np.asarray(points)
    return points[..., 0], points[..., 1], points[..., 2]


def skin_shell_trace(start_x, stop_x, radius, color, name, axis, angle=0.0):
    """Create a translucent wireframe cylinder rigidly mounted to a limb."""
    lines = []
    phi = np.linspace(0.0, 2.0 * np.pi, 49)
    for axial in np.linspace(start_x, stop_x, 10):
        lines.extend(np.column_stack((np.full_like(phi, axial),
                                      radius * np.cos(phi),
                                      radius * np.sin(phi))).tolist())
        lines.append([np.nan, np.nan, np.nan])
    for phase in np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False):
        axial = np.linspace(start_x, stop_x, 24)
        lines.extend(np.column_stack((axial, np.full_like(axial, radius * np.cos(phase)),
                                      np.full_like(axial, radius * np.sin(phase)))).tolist())
        lines.append([np.nan, np.nan, np.nan])
    points = rotate_points(lines, axis, angle)
    x_value, y_value, z_value = display_xyz(points)
    return go.Scatter3d(
        x=x_value, y=y_value, z=z_value, mode="lines", name=name,
        line={"color": color, "width": 2}, opacity=0.32,
        hovertemplate=f"{name}<extra></extra>",
    )


def tapered_limb_trace(start_x, stop_x, color, name, axis, angle=0.0):
    """Create the translucent tapered 3D limb body used by the old viewer."""
    outer = max(abs(start_x), abs(stop_x))
    vertices = []
    for x_value in (start_x, stop_x):
        scale = abs(x_value) / outer
        half_y = 0.035 + 0.030 * scale
        half_z = 0.055 + 0.045 * scale
        for y_value, z_value in ((-half_y, -half_z), (half_y, -half_z),
                                 (half_y, half_z), (-half_y, half_z)):
            vertices.append([x_value, y_value, z_value])
    vertices = rotate_points(vertices, axis, angle)
    x_value, y_value, z_value = display_xyz(vertices)
    triangles = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                 (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                 (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0))
    return go.Mesh3d(
        x=x_value, y=y_value, z=z_value,
        i=[item[0] for item in triangles], j=[item[1] for item in triangles],
        k=[item[2] for item in triangles], color=color, opacity=0.42,
        flatshading=True, name=name,
        lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.3, "roughness": 0.5},
        hovertemplate=f"{name}<extra></extra>",
    )


def bearing_trace(axis, radius, half_length):
    """Create a translucent cylindrical revolute-joint bearing."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    helper = np.asarray([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    radial_a = np.cross(axis, helper)
    radial_a /= np.linalg.norm(radial_a)
    radial_b = np.cross(axis, radial_a)
    phi = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    rings = []
    for axial in (-half_length, half_length):
        rings.append(axial * axis + radius * (np.cos(phi)[:, None] * radial_a
                                               + np.sin(phi)[:, None] * radial_b))
    vertices = np.vstack(rings)
    x_value, y_value, z_value = display_xyz(vertices)
    triangles = []
    count = len(phi)
    for index in range(count):
        nxt = (index + 1) % count
        triangles.extend(((index, nxt, count + nxt), (index, count + nxt, count + index)))
    return go.Mesh3d(
        x=x_value, y=y_value, z=z_value,
        i=[item[0] for item in triangles], j=[item[1] for item in triangles],
        k=[item[2] for item in triangles], color="#aeb5bd", opacity=0.55,
        flatshading=True, name="revolute joint",
        hovertemplate="Revolute joint bearing<extra></extra>",
    )


def build_html(dataset, topology, torque, stiffness, output, max_frames,
               visual_relaxation_steps, frame_duration_ms=80):
    periods, samples = torque.shape
    global_time = np.concatenate([
        dataset["t"][p] + p * dataset["period_seconds"] for p in range(periods)
    ])
    theta = dataset["theta"].reshape(-1)
    target = dataset["target"].reshape(-1)
    spring_torque = torque.reshape(-1)
    motor = target - spring_torque
    profile_index = np.asarray(
        dataset.get("profile_index", np.zeros(periods, dtype=int)), dtype=int
    )
    period_in_profile = np.asarray(
        dataset.get("period_in_profile", np.arange(periods)), dtype=int
    )
    profile_count = int(dataset.get("profile_count", 1))
    periods_per_profile = int(dataset.get("periods_per_profile", periods))
    schedule = np.broadcast_to(
        stiffness[:, None, :], (periods, samples, stiffness.shape[1])
    ).reshape(-1, stiffness.shape[1])
    frame_indices = np.unique(np.linspace(0, len(global_time) - 1, min(max_frames, len(global_time))).astype(int))
    device = topology["local_positions"].device
    prescribed = prescribed_positions(
        topology, torch.as_tensor(theta[frame_indices], dtype=torch.float32, device=device)
    )
    positions = relax_positions(
        topology, prescribed,
        torch.as_tensor(schedule[frame_indices].copy(), dtype=torch.float32, device=device),
        steps=visual_relaxation_steps,
    ).detach().cpu().numpy()
    spring_a = topology["spring_a"].detach().cpu().numpy()
    spring_b = topology["spring_b"].detach().cpu().numpy()
    kinds = np.asarray(topology["node_types"])
    node_palette = {"skin1": "#2864b7", "skin2": "#15956f", "internal": "#f4a261",
                    "limb1": "#24577a", "limb2": "#16a085"}
    node_colors = [node_palette.get(kind, "#7570b3") for kind in kinds]
    node_symbols = ["square" if kind in ("skin1", "skin2") else "circle" for kind in kinds]
    node_sizes = [9 if kind == "internal" else 8 for kind in kinds]

    figure = make_subplots(
        rows=2, cols=2, specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                               [None, {"type": "xy"}]],
        column_widths=[0.5, 0.5], subplot_titles=("3D adaptive spring joint", "Torque history", "Current stiffness vector"),
    )
    color_min = max(float(np.min(stiffness)), 1e-6)
    color_max = max(float(np.max(stiffness)), color_min + 1e-6)
    initial_colors = stiffness_colors(stiffness[0], color_min, color_max)
    spring_data = topology["data"].get("springs", [])

    # Draw the physical members underneath the spring network using the
    # translucent 3D treatment from the original topology viewer.
    limb_length = max(float(np.max(np.abs(topology["local_positions"][:, 0].detach().cpu().numpy()))) + 0.2, 1.0)
    joint_axis = topology["joint_axis"].detach().cpu().numpy()
    figure.add_trace(tapered_limb_trace(
        -limb_length, -0.065, "#31688e", "fixed limb", joint_axis,
    ), row=1, col=1)
    moving_limb_trace_index = len(figure.data)
    figure.add_trace(tapered_limb_trace(
        0.065, limb_length, "#20a486", "moving limb", joint_axis,
        theta[frame_indices[0]],
    ), row=1, col=1)
    figure.add_trace(bearing_trace(
        joint_axis, float(topology["data"].get("bearing_radius", 0.12)),
        float(topology["data"].get("bearing_half_length", 0.13)),
    ), row=1, col=1)
    skin_radius = float(topology["data"].get("skin_radius", 0.74))
    figure.add_trace(skin_shell_trace(
        -limb_length, -0.15, skin_radius, "#2864b7", "fixed skin shell", joint_axis,
    ), row=1, col=1)
    moving_shell_trace_index = len(figure.data)
    figure.add_trace(skin_shell_trace(
        0.15, limb_length, skin_radius, "#15956f", "moving skin shell", joint_axis,
        theta[frame_indices[0]],
    ), row=1, col=1)
    spring_trace_indices = []
    for spring_index, (a, b) in enumerate(zip(spring_a, spring_b)):
        label = (spring_data[spring_index].get("name", f"spring {spring_index}")
                 if spring_index < len(spring_data) else f"spring {spring_index}")
        spring_trace_indices.append(len(figure.data))
        figure.add_trace(go.Scatter3d(
            x=[positions[0, a, 0], positions[0, b, 0]],
            y=[positions[0, a, 1], positions[0, b, 1]],
            z=[positions[0, a, 2], positions[0, b, 2]],
            mode="lines", line={"color": initial_colors[spring_index], "width": 5},
            name="springs", legendgroup="springs", showlegend=spring_index == 0,
            customdata=np.full((2, 1), stiffness[0, spring_index]),
            hovertemplate=f"{label}<br>stiffness: %{{customdata[0]:.1f}} N/m<extra></extra>",
        ), row=1, col=1)
    node_trace_index = len(figure.data)
    figure.add_trace(go.Scatter3d(x=positions[0, :, 0], y=positions[0, :, 1], z=positions[0, :, 2],
                                  mode="markers", marker={"size": node_sizes, "color": node_colors,
                                                          "symbol": node_symbols,
                                  "line": {"color": "white", "width": 2}},
                                  name="nodes", text=topology["names"], hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    figure.add_trace(go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None], mode="markers",
        marker={"size": 0.01, "color": [np.log10(color_min), np.log10(color_max)],
                "cmin": np.log10(color_min), "cmax": np.log10(color_max),
                "colorscale": "Turbo", "showscale": True,
                "colorbar": {"title": "Stiffness [N/m]", "x": 0.46,
                             "tickvals": [np.log10(color_min), np.log10(color_max)],
                             "ticktext": [f"{color_min:.1f}", f"{color_max:.1f}"]}},
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    figure.add_trace(go.Scatter(x=global_time, y=target, name="target", line={"color": "black", "dash": "dash"}), row=1, col=2)
    figure.add_trace(go.Scatter(x=global_time, y=spring_torque, name="spring", line={"color": "#2a8c62"}), row=1, col=2)
    figure.add_trace(go.Scatter(x=global_time, y=motor, name="residual motor", line={"color": "#c44e52"}), row=1, col=2)
    cursor_y = [min(target.min(), spring_torque.min(), motor.min()), max(target.max(), spring_torque.max(), motor.max())]
    cursor_trace_index = len(figure.data)
    figure.add_trace(go.Scatter(x=[global_time[frame_indices[0]]] * 2, y=cursor_y,
                                mode="lines", line={"color": "#f2a900", "width": 3},
                                name="current time"), row=1, col=2)
    bar_trace_index = len(figure.data)
    figure.add_trace(go.Bar(
        x=np.arange(stiffness.shape[1]), y=schedule[frame_indices[0]],
        marker_color=initial_colors, name="spring stiffness", showlegend=True,
    ), row=2, col=2)
    # Keep each subplot's labels local instead of mixing every trace into the
    # model legend at the left of the figure.
    figure.update_traces(legend="legend", row=1, col=1)
    figure.update_traces(legend="legend2", row=1, col=2)
    figure.update_traces(legend="legend3", row=2, col=2)
    frames = []
    for frame_number, index in enumerate(frame_indices):
        period = min(index // samples, periods - 1)
        if period == 0:
            policy_label = "default stiffness"
        elif period_in_profile[period] == 0:
            policy_label = "profile switch: stiffness from preceding profile"
        else:
            policy_label = "updated at period boundary"
        colors = stiffness_colors(stiffness[period], color_min, color_max)
        frame_data = []
        frame_traces = []
        frame_data.append(tapered_limb_trace(
            0.065, limb_length, "#20a486", "moving limb", joint_axis, theta[index],
        ))
        frame_traces.append(moving_limb_trace_index)
        frame_data.append(skin_shell_trace(
            0.15, limb_length, skin_radius, "#15956f", "moving skin shell",
            joint_axis, theta[index],
        ))
        frame_traces.append(moving_shell_trace_index)
        for spring_index, (a, b) in enumerate(zip(spring_a, spring_b)):
            frame_data.append(go.Scatter3d(
                x=[positions[frame_number, a, 0], positions[frame_number, b, 0]],
                y=[positions[frame_number, a, 1], positions[frame_number, b, 1]],
                z=[positions[frame_number, a, 2], positions[frame_number, b, 2]],
                line={"color": colors[spring_index], "width": 5},
                customdata=np.full((2, 1), stiffness[period, spring_index]),
            ))
            frame_traces.append(spring_trace_indices[spring_index])
        frame_data.extend([
            go.Scatter3d(x=positions[frame_number, :, 0], y=positions[frame_number, :, 1], z=positions[frame_number, :, 2]),
            go.Scatter(x=[global_time[index]] * 2, y=cursor_y),
            go.Bar(x=np.arange(stiffness.shape[1]), y=stiffness[period], marker_color=colors),
        ])
        frame_traces.extend([node_trace_index, cursor_trace_index, bar_trace_index])
        frames.append(go.Frame(
            name=str(frame_number),
            data=frame_data,
            traces=frame_traces,
            layout=go.Layout(title_text=(
                f"Profile {profile_index[period] + 1}/{profile_count} | "
                f"period {period_in_profile[period] + 1}/{periods_per_profile} | "
                f"global period {period + 1}/{periods} | t={global_time[index]:.2f} s | "
                f"{policy_label}"
            )),
        ))
    figure.frames = frames
    buttons = [{"label": "Play", "method": "animate",
                "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                                 "fromcurrent": True}]},
               {"label": "Pause", "method": "animate",
                "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}]
    steps = [{"label": str(i + 1), "method": "animate",
              "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
             for i in range(len(frames))]
    figure.update_layout(
        height=760, template="plotly_white", title=frames[0].layout.title.text,
        updatemenus=[{"type": "buttons", "buttons": buttons, "x": 0.52, "y": 0.02}],
        sliders=[{"steps": steps, "x": 0.05, "len": 0.4, "currentvalue": {"prefix": "Frame "}}],
        scene={"aspectmode": "data", "xaxis_title": "X [m]",
               "yaxis_title": "Z [m]", "zaxis_title": "Y [m]",
               "camera": {"eye": {"x": 1.55, "y": -1.7, "z": 1.05}}},
        legend={"title": {"text": "3D model"}, "x": 0.01, "y": 0.98,
                "xanchor": "left", "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.78)"},
        legend2={"title": {"text": "Torque"}, "x": 0.99, "y": 0.98,
                 "xanchor": "right", "yanchor": "top",
                 "bgcolor": "rgba(255,255,255,0.78)"},
        legend3={"title": {"text": "Stiffness"}, "x": 0.99, "y": 0.43,
                 "xanchor": "right", "yanchor": "top",
                 "bgcolor": "rgba(255,255,255,0.78)"},
    )
    figure.update_xaxes(title_text="Time [s]", row=1, col=2)
    figure.update_yaxes(title_text="Torque [N m]", row=1, col=2)
    figure.update_xaxes(title_text="Spring index", row=2, col=2)
    figure.update_yaxes(title_text="Stiffness [N/m]", row=2, col=2)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, auto_play=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument(
        "--periods", type=int, default=6,
        help="Periods per generated torque-angle profile.",
    )
    parser.add_argument(
        "--demo-profiles", type=int, default=1,
        help="Number of distinct generated profiles shown consecutively.",
    )
    parser.add_argument("--relaxation-steps", type=int, default=300)
    parser.add_argument("--visual-relaxation-steps", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--frame-duration-ms", type=int, default=80,
        help="Playback duration assigned to each interactive frame.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.periods < 1 or args.demo_profiles < 1 or args.frame_duration_ms < 1:
        parser.error("periods, demo-profiles, and frame-duration-ms must be positive")
    model, metadata = load_checkpoint(args.checkpoint)
    topology = load_spatial_topology(args.topology, torch.device(args.device))
    angles = np.radians(ANGLE_DEGREES)
    basis = spatial_initial_basis(topology, angles, args.relaxation_steps)
    dataset = (
        dataset_from_csv(args.input_csv, metadata, angles, basis)
        if args.input_csv else demo_dataset(
            metadata, angles, basis, args.periods, args.seed, args.demo_profiles
        )
    )
    torque, stiffness, _ = deploy(
        model, metadata, dataset, topology, args.relaxation_steps, 1024, 0
    )
    build_html(dataset, topology, torque, stiffness, args.output,
               args.max_frames, args.visual_relaxation_steps, args.frame_duration_ms)
    print(f"Saved interactive simulation to {args.output}")


if __name__ == "__main__":
    main()
