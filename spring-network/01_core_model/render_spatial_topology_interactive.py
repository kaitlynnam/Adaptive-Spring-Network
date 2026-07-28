"""Export an animated, standalone Plotly viewer for the spatial spring joint."""

from pathlib import Path
import argparse
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_core_model"))
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from mechanics_3d import load_spatial_topology, prescribed_positions, relax_positions
from profile_generator import generate_classified_profile_parameters
from train_adaptive_3d import causal_spatial_rollout, initial_basis
from train_adaptive_dataset import build_dataset


DEFAULT_TOPOLOGY = (
    PROJECT_ROOT / "topologies" / "spatial"
    / "surface_search" / "candidate_022.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "plots" / "current" / "spatial"
    / "candidate022_48spring_dynamic_demo.html"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "spatial" / "surface_candidate_022_screen.npz"
)


def learned_demo_rollout(
    topology,
    checkpoint_path,
    profile_index=0,
    demo_frequency_hz=None,
    use_heldout_profile=False,
):
    """Generate one deterministic torque profile and run the saved controller."""
    with np.load(checkpoint_path, allow_pickle=True) as saved:
        model = {name: saved[name] for name in ("w1", "b1", "w2", "b2")}
        samples = int(saved["samples"])
        duration = float(saved["duration"])
        window_size = int(saved["window_size"])
        seed = int(saved["seed"])
        motion_mode = str(saved["motion_mode"])
        fixed_frequency_hz = float(saved["fixed_frequency_hz"])
        min_k, max_k = float(saved["min_k"]), float(saved["max_k"])
        relaxation_steps = int(saved["relaxation_steps"])
        profiles_per_family = int(saved["profiles_per_family"])
        test_profiles_per_family = int(saved["test_profiles_per_family"])
        scales = {
            "theta": float(saved["theta_scale"]),
            "theta_dot": float(saved["theta_dot_scale"]),
            "theta_ddot": float(saved["theta_ddot_scale"]),
            "torque": float(saved["torque_scale"]),
            "window_size": window_size,
        }
    if demo_frequency_hz is not None:
        fixed_frequency_hz = float(demo_frequency_hz)
    if use_heldout_profile:
        rng = np.random.default_rng(seed)
        generate_classified_profile_parameters(rng, profiles_per_family)
        profiles = generate_classified_profile_parameters(
            rng, test_profiles_per_family,
        )
        if not 0 <= profile_index < len(profiles):
            raise IndexError("held-out profile index is out of range")
        params = [profiles[profile_index]]
        dataset_seed = seed + 30_000 + profile_index
    else:
        rng = np.random.default_rng(seed + 91_000)
        profiles = generate_classified_profile_parameters(rng, profile_index + 1)
        params = [profiles[profile_index]]
        dataset_seed = seed + 92_000
    angles = np.radians(np.arange(-45.0, 46.0, 5.0))
    basis = initial_basis(topology, angles, relaxation_steps)
    dataset = build_dataset(
        params,
        angles_rad=angles,
        basis_by_angle=basis,
        duration=duration,
        samples=samples,
        window_size=window_size,
        scales=scales,
        seed=dataset_seed,
        stiffness_update_mode="timestep",
        include_profile_descriptor=False,
        motion_mode=motion_mode,
        fixed_frequency_hz=fixed_frequency_hz,
    )
    predicted, stiffness, residual = causal_spatial_rollout(
        model, dataset, topology, min_k, max_k, relaxation_steps
    )
    return dataset, predicted, stiffness, residual, relaxation_steps


def interpolate_rollout(dataset, spring_torque, stiffness, max_angle_step_deg):
    """Interpolate a causal rollout so no rendered joint step exceeds the limit."""
    theta = np.asarray(dataset["theta"])
    segments = []
    for index in range(len(theta) - 1):
        delta_deg = abs(np.degrees(theta[index + 1] - theta[index]))
        count = max(1, int(np.ceil(delta_deg / max_angle_step_deg)))
        segments.extend((index, fraction / count) for fraction in range(count))
    segments.append((len(theta) - 1, 0.0))
    left = np.asarray([item[0] for item in segments], dtype=int)
    alpha = np.asarray([item[1] for item in segments], dtype=float)
    right = np.minimum(left + 1, len(theta) - 1)

    def blend(values):
        values = np.asarray(values)
        shape = (len(alpha),) + (1,) * (values.ndim - 1)
        weight = alpha.reshape(shape)
        return values[left] * (1.0 - weight) + values[right] * weight

    interpolated = dict(dataset)
    for key in ("t", "theta", "theta_dot", "theta_ddot", "target"):
        interpolated[key] = blend(dataset[key])
    interpolated["samples_per_profile"] = len(left)
    return interpolated, blend(spring_torque), blend(stiffness)


def line_coordinates(positions, spring_a, spring_b):
    x, y, z = [], [], []
    for a, b in zip(spring_a, spring_b):
        x.extend((positions[a, 0], positions[b, 0], None))
        y.extend((positions[a, 1], positions[b, 1], None))
        z.extend((positions[a, 2], positions[b, 2], None))
    return x, y, z


def spring_traces(topology, positions, stiffness, color_min, color_max):
    """Draw springs separately so color and hover text encode stiffness."""
    traces = []
    log_min = np.log10(max(color_min, 1e-6))
    log_max = np.log10(max(color_max, color_min + 1e-6))
    span = max(log_max - log_min, 1e-9)
    spring_a = topology["spring_a"].detach().cpu().numpy()
    spring_b = topology["spring_b"].detach().cpu().numpy()
    spring_data = topology["data"].get("springs", [])
    for index, (a, b, value) in enumerate(zip(spring_a, spring_b, stiffness)):
        fraction = float(np.clip((np.log10(max(value, 1e-6)) - log_min) / span, 0.0, 1.0))
        color = sample_colorscale("Turbo", [fraction])[0]
        label = (
            spring_data[index].get("name", f"spring {index}")
            if index < len(spring_data) else f"spring {index}"
        )
        traces.append(go.Scatter3d(
            x=[positions[a, 0], positions[b, 0]],
            y=[positions[a, 1], positions[b, 1]],
            z=[positions[a, 2], positions[b, 2]],
            mode="lines", name="springs",
            legendgroup="springs", showlegend=index == 0,
            line={"color": color, "width": 5},
            customdata=np.full((2, 1), value),
            hovertemplate=(
                f"{label}<br>configured stiffness: "
                "%{customdata[0]:.1f} N/m<extra></extra>"
            ),
        ))
    return traces


def stiffness_colorbar(color_min, color_max):
    tick_values = np.asarray([1, 10, 50, 100, 200, 400, 800], dtype=float)
    tick_values = tick_values[
        (tick_values >= color_min) & (tick_values <= color_max * 1.001)
    ]
    log_min = np.log10(max(color_min, 1e-6))
    log_max = np.log10(max(color_max, color_min + 1e-6))
    return go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None],
        mode="markers", showlegend=False, hoverinfo="skip",
        marker={
            "size": 0.01, "color": [log_min, log_max],
            "cmin": log_min, "cmax": log_max,
            "colorscale": "Turbo", "showscale": True,
            "colorbar": {
                "title": {"text": "stiffness<br>[N/m]"},
                "thickness": 18,
                "tickvals": np.log10(tick_values).tolist(),
                "ticktext": [f"{value:g}" for value in tick_values],
            },
        },
    )


def wire_cylinder(start_x, stop_x, radius, angle=0.0):
    x, y, z = [], [], []
    for axial in np.linspace(start_x, stop_x, 9):
        phi = np.linspace(0.0, 2.0 * np.pi, 49)
        points = np.column_stack((
            np.full_like(phi, axial), radius * np.cos(phi), radius * np.sin(phi)
        ))
        if angle:
            c, s = np.cos(angle), np.sin(angle)
            px, pz = points[:, 0].copy(), points[:, 2].copy()
            points[:, 0], points[:, 2] = c * px + s * pz, -s * px + c * pz
        x.extend((*points[:, 0], None))
        y.extend((*points[:, 1], None))
        z.extend((*points[:, 2], None))
    for phi in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        points = np.column_stack((
            np.linspace(start_x, stop_x, 25),
            np.full(25, radius * np.cos(phi)),
            np.full(25, radius * np.sin(phi)),
        ))
        if angle:
            c, s = np.cos(angle), np.sin(angle)
            px, pz = points[:, 0].copy(), points[:, 2].copy()
            points[:, 0], points[:, 2] = c * px + s * pz, -s * px + c * pz
        x.extend((*points[:, 0], None))
        y.extend((*points[:, 1], None))
        z.extend((*points[:, 2], None))
    return x, y, z


def skin_shell_trace(start_x, stop_x, radius, color, name, angle=0.0):
    """Low-poly translucent shell rigidly attached to one limb segment."""
    phi = np.linspace(0.0, 2.0 * np.pi, 25)[:-1]
    vertices = np.vstack([
        np.column_stack((
            np.full_like(phi, start_x), radius * np.cos(phi), radius * np.sin(phi)
        )),
        np.column_stack((
            np.full_like(phi, stop_x), radius * np.cos(phi), radius * np.sin(phi)
        )),
    ])
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        px, pz = vertices[:, 0].copy(), vertices[:, 2].copy()
        vertices[:, 0], vertices[:, 2] = c * px + s * pz, -s * px + c * pz
    count = len(phi)
    triangles = []
    for index in range(count):
        nxt = (index + 1) % count
        triangles.extend(((index, nxt, count + nxt), (index, count + nxt, count + index)))
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=[tri[0] for tri in triangles],
        j=[tri[1] for tri in triangles],
        k=[tri[2] for tri in triangles],
        name=name, color=color, opacity=0.13, flatshading=True,
        hovertemplate=f"{name}<br>rigid limb-mounted skin shell<extra></extra>",
    )


def node_trace(positions, indices, name, color, symbol, size):
    points = positions[indices]
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers", name=name,
        marker={"color": color, "size": size, "symbol": symbol,
                "line": {"color": "white", "width": 1}},
        hovertemplate=f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
    )


def tapered_limb_trace(start_x, stop_x, color, name, angle=0.0):
    """Solid tapered rectangular limb mesh, rotated about the y-axis."""
    outer = max(abs(start_x), abs(stop_x))
    vertices = []
    for x in (start_x, stop_x):
        scale = abs(x) / outer
        half_y = 0.035 + 0.030 * scale
        half_z = 0.055 + 0.045 * scale
        for y, z in (
            (-half_y, -half_z), (half_y, -half_z),
            (half_y, half_z), (-half_y, half_z),
        ):
            vertices.append([x, y, z])
    vertices = np.asarray(vertices, dtype=float)
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        x, z = vertices[:, 0].copy(), vertices[:, 2].copy()
        vertices[:, 0], vertices[:, 2] = c * x + s * z, -s * x + c * z
    triangles = (
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    )
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=[item[0] for item in triangles],
        j=[item[1] for item in triangles],
        k=[item[2] for item in triangles],
        name=name, color=color, opacity=0.88, flatshading=True,
        lighting={
            "ambient": 0.34, "diffuse": 0.88, "specular": 0.38,
            "roughness": 0.48, "fresnel": 0.12,
        },
        lightposition={"x": 1.8, "y": -2.2, "z": 2.6},
        contour={"show": True, "color": "rgba(255,255,255,0.72)", "width": 2},
        hovertemplate=f"{name}<extra></extra>",
    )


def support_trace(topology, positions, angle):
    """Visible rigid load paths from joint nodes to their corresponding limb."""
    local_positions = topology["local_positions"].detach().cpu().numpy()
    x_values, y_values, z_values = [], [], []
    if topology["data"].get("joint_nodes_on_limb_surface", False):
        return go.Scatter3d(
            x=[], y=[], z=[], mode="lines", name="rigid node mounting webs",
            visible=False, showlegend=False, hoverinfo="skip",
        )
    for index, kind in enumerate(topology["node_types"]):
        if kind not in ("limb1", "limb2"):
            continue
        local_node = local_positions[index]
        outer = 0.98 if kind == "limb1" else 1.11
        scale = abs(local_node[0]) / outer
        half_y = 0.035 + 0.030 * scale
        half_z = 0.055 + 0.045 * scale
        ellipse = np.sqrt(
            (local_node[1] / half_y) ** 2 + (local_node[2] / half_z) ** 2
        )
        base = local_node.copy()
        if ellipse > 1.0:
            base[1:] /= ellipse
        if kind == "limb2":
            c, s = np.cos(angle), np.sin(angle)
            bx, bz = base[0], base[2]
            base[0], base[2] = c * bx + s * bz, -s * bx + c * bz
        node = positions[index]
        x_values.extend((base[0], node[0], None))
        y_values.extend((base[1], node[1], None))
        z_values.extend((base[2], node[2], None))
    return go.Scatter3d(
        x=x_values, y=y_values, z=z_values, mode="lines",
        name="rigid node mounting webs",
        line={"color": "#707780", "width": 4},
        hovertemplate="6 mm-thick rigid mounting web<extra></extra>",
    )


def dynamic_traces(topology, positions, angle, stiffness, color_min, color_max):
    kinds = topology["node_types"]
    groups = {
        "skin1": [i for i, kind in enumerate(kinds) if kind == "skin1"],
        "skin2": [i for i, kind in enumerate(kinds) if kind == "skin2"],
        "internal": [i for i, kind in enumerate(kinds) if kind == "internal"],
        "limb1": [i for i, kind in enumerate(kinds) if kind == "limb1"],
        "limb2": [i for i, kind in enumerate(kinds) if kind == "limb2"],
    }
    radius = topology["data"].get("skin_radius", 0.74)
    return [
        *spring_traces(topology, positions, stiffness, color_min, color_max),
        node_trace(positions, groups["skin1"], "proximal skin anchors", "#2864b7", "square", 4),
        node_trace(positions, groups["skin2"], "distal skin anchors", "#15956f", "square", 4),
        node_trace(positions, groups["internal"], "free internal nodes", "#f4a261", "circle", 7),
        node_trace(positions, groups["limb1"], "proximal joint nodes", "#24577a", "circle", 5),
        node_trace(positions, groups["limb2"], "distal joint nodes", "#16a085", "circle", 5),
        tapered_limb_trace(-0.98, -0.065, "#31688e", "proximal solid limb"),
        tapered_limb_trace(0.065, 1.11, "#20a486", "distal solid limb", angle),
        support_trace(topology, positions, angle),
        skin_shell_trace(
            -1.05, -0.15, radius, "#4c78a8", "proximal skin shell"
        ),
        skin_shell_trace(
            0.15, 1.12, radius, "#36a57a", "distal skin shell", angle
        ),
    ]


def bearing_trace(topology):
    radius = topology["bearing_radius"]
    half = topology["bearing_half_length"]
    phi, y = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, 40),
        np.linspace(-half, half, 12),
    )
    return go.Surface(
        x=radius * np.cos(phi), y=y, z=radius * np.sin(phi),
        name="bearing", showscale=False, opacity=0.92,
        colorscale=[[0, "#b9bdc2"], [1, "#e2e4e7"]],
        hoverinfo="skip",
    )


def render(topology_path, output_path, step=1, relaxation_steps=300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topology = load_spatial_topology(topology_path, device)
    angles_deg = np.arange(-45, 46, step, dtype=float)
    theta = torch.as_tensor(
        np.radians(angles_deg), dtype=torch.float32, device=device
    )
    stiffness = topology["initial_stiffness"].unsqueeze(0).repeat(len(theta), 1)
    positions = relax_positions(
        topology, prescribed_positions(topology, theta), stiffness,
        steps=relaxation_steps,
    ).detach().cpu().numpy()

    stiffness_np = stiffness.detach().cpu().numpy()
    color_min = float(np.min(stiffness_np))
    color_max = float(np.max(stiffness_np))
    initial = dynamic_traces(
        topology, positions[0], float(theta[0].cpu()),
        stiffness_np[0], color_min, color_max,
    )
    dynamic_count = len(initial)
    figure = go.Figure(data=[
        *initial,
        bearing_trace(topology),
        stiffness_colorbar(color_min, color_max),
    ])
    figure.frames = [
        go.Frame(
            name=f"{angle:+.0f}",
            data=dynamic_traces(
                topology, state, np.radians(angle),
                frame_stiffness, color_min, color_max,
            ),
            traces=list(range(dynamic_count)),
        )
        for angle, state, frame_stiffness in zip(
            angles_deg, positions, stiffness_np
        )
    ]
    slider_steps = [
        {
            "label": f"{angle:+.0f}°",
            "method": "animate",
            "args": [[f"{angle:+.0f}"], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        }
        for angle in angles_deg
    ]
    figure.update_layout(
        title=f"{topology['data']['name']} — fully relaxed 3D mechanics",
        template="plotly_white",
        scene={
            "xaxis": {"title": "x [m]", "range": [-1.15, 1.20]},
            "yaxis": {"title": "y [m]", "range": [-0.95, 0.95]},
            "zaxis": {"title": "z [m]", "range": [-0.95, 0.95]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.3, "y": 1.0, "z": 1.0},
            "camera": {
                "eye": {"x": 1.55, "y": 1.15, "z": -1.65},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            },
            "dragmode": "orbit",
            "uirevision": "preserve-user-camera",
        },
        legend={"groupclick": "toggleitem"},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        annotations=[{
            "text": (
                "Drag to orbit • Shift+drag to pan • Scroll to zoom • "
                "Double-click to reset the view"
            ),
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 1.0,
            "xanchor": "center", "yanchor": "bottom", "showarrow": False,
            "font": {"size": 13, "color": "#444"},
        }],
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.02, "y": 0.02,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [None, {
                    "fromcurrent": True,
                    "frame": {"duration": 70, "redraw": True},
                    "transition": {"duration": 0},
                }]},
                {"label": "❚❚ Pause", "method": "animate", "args": [[None], {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                }]},
            ],
        }],
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "joint angle: "},
            "pad": {"t": 35}, "steps": slider_steps,
        }],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path, include_plotlyjs=True, full_html=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )
    print(f"device={device}")
    print(output_path)


def render_dynamic(
    topology_path, output_path, checkpoint_path, profile_index=0,
    max_angle_step_deg=0.3, demo_frequency_hz=0.25,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topology = load_spatial_topology(topology_path, device)
    dataset, spring_torque, stiffness_np, residual, relaxation_steps = (
        learned_demo_rollout(
            topology, checkpoint_path, profile_index, demo_frequency_hz
        )
    )
    dataset, spring_torque, stiffness_np = interpolate_rollout(
        dataset, spring_torque, stiffness_np, max_angle_step_deg
    )
    time, target_torque = dataset["t"], dataset["target"]
    theta = torch.as_tensor(dataset["theta"], dtype=torch.float32, device=device)
    stiffness = torch.as_tensor(stiffness_np, dtype=torch.float32, device=device)
    positions = relax_positions(
        topology, prescribed_positions(topology, theta), stiffness,
        steps=relaxation_steps,
    ).detach().cpu().numpy()
    color_min, color_max = float(np.min(stiffness_np)), float(np.max(stiffness_np))
    initial = dynamic_traces(
        topology, positions[0], float(theta[0].cpu()),
        stiffness_np[0], color_min, color_max,
    )
    dynamic_count = len(initial)
    figure = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.72, 0.28], horizontal_spacing=0.025,
    )
    for trace in initial:
        figure.add_trace(trace, row=1, col=1)
    figure.add_trace(bearing_trace(topology), row=1, col=1)
    figure.add_trace(stiffness_colorbar(color_min, color_max), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=time, y=target_torque, mode="lines", name="target torque",
        line={"color": "#222222", "width": 2, "dash": "dash"},
    ), row=1, col=2)
    figure.add_trace(go.Scatter(
        x=time, y=spring_torque, mode="lines", name="spring torque",
        line={"color": "#277da1", "width": 2},
    ), row=1, col=2)
    cursor_index = len(figure.data)
    figure.add_trace(go.Scatter(
        x=[time[0]], y=[target_torque[0]], mode="markers", name="current time",
        marker={"color": "#e63946", "size": 11},
    ), row=1, col=2)
    figure.frames = [
        go.Frame(
            name=str(index),
            data=[
                *dynamic_traces(
                    topology, state, float(frame_theta),
                    frame_stiffness, color_min, color_max,
                ),
                go.Scatter(
                    x=[frame_time], y=[frame_target], mode="markers",
                    marker={"color": "#e63946", "size": 11},
                ),
            ],
            traces=[*range(dynamic_count), cursor_index],
        )
        for index, (frame_time, frame_target, frame_theta, state, frame_stiffness)
        in enumerate(zip(
            time, target_torque, dataset["theta"], positions, stiffness_np
        ))
    ]
    slider_steps = [{
        "label": f"{frame_time:.2f}s", "method": "animate",
        "args": [[str(index)], {
            "mode": "immediate", "frame": {"duration": 0, "redraw": True},
            "transition": {"duration": 0},
        }],
    } for index, frame_time in enumerate(time)]
    figure.update_xaxes(title_text="time [s]", row=1, col=2)
    figure.update_yaxes(title_text="torque [N·m]", row=1, col=2)
    figure.update_layout(
        title={
            "text": (
                f"{topology['data']['name']} — learned stiffness and relaxed 3D "
                f"mechanics at {demo_frequency_hz:.2f} Hz | max equilibrium "
                f"residual {np.max(residual):.3f} N"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
            "yanchor": "top",
        },
        template="plotly_white",
        scene={
            "xaxis": {"title": "x [m]", "range": [-1.15, 1.20]},
            "yaxis": {"title": "y [m]", "range": [-0.95, 0.95]},
            "zaxis": {"title": "z [m]", "range": [-0.95, 0.95]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.3, "y": 1.0, "z": 1.0},
            "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 1.15}},
            "dragmode": "orbit", "uirevision": "preserve-user-camera",
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.13,
            "yanchor": "bottom",
            "groupclick": "toggleitem",
            "bgcolor": "rgba(255,255,255,0.88)",
        },
        margin={"l": 0, "r": 25, "t": 165, "b": 10},
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.02, "y": 0.02,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {
                    "fromcurrent": True,
                    "frame": {"duration": 10, "redraw": True},
                    "transition": {"duration": 8, "easing": "linear"},
                }]},
                {"label": "Pause", "method": "animate", "args": [[None], {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                }]},
            ],
        }],
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "time: "},
            "pad": {"t": 35}, "steps": slider_steps,
        }],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path, include_plotlyjs=True, full_html=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )
    print(f"device={device}")
    print(f"profile_samples={len(time)}")
    print(f"maximum_rendered_angle_step={max_angle_step_deg:.3f} deg")
    print(f"demo_frequency={demo_frequency_hz:.3f} Hz")
    print(f"dynamic_stiffness_range={color_min:.3f}..{color_max:.3f} N/m")
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument("--max-angle-step-deg", type=float, default=0.3)
    parser.add_argument("--demo-frequency-hz", type=float, default=0.25)
    args = parser.parse_args()
    render_dynamic(
        args.topology, args.output, args.checkpoint, args.profile_index,
        args.max_angle_step_deg, args.demo_frequency_hz,
    )


if __name__ == "__main__":
    main()
