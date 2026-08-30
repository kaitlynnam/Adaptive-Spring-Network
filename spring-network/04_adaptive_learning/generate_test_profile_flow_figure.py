"""Create a publication figure tracing one real test profile through the network."""

from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "tables" / "period_adaptive_3d" /
    "period_adaptive_3d_collision_exact5refresh_6x1000_seed101_"
    "1000profiles_6periods_complete_data.npz"
)
OUTPUT_PATH = PROJECT_ROOT / "plots" / "period_adaptive_3d" / "fig03_test_profile_flow.png"
ASSET_DIR = PROJECT_ROOT / "plots" / "period_adaptive_3d" / "fig01_powerpoint_assets"
FIGURE_ONE_PATH = PROJECT_ROOT / "plots" / "period_adaptive_3d" / "fig01_topology.png"
TOPOLOGY_PATH = PROJECT_ROOT / "topologies" / "spatial" / "hybrid_internal_skin_3d_60_spring.json"
SELECTED_PROFILE_NUMBER = 629


BLUE = "#2457a6"
GREEN = "#188b67"
ORANGE = "#e76f51"
DARK = "#263238"
GRID = "#d9e0e5"
PALE = "#eef3f7"


def choose_representative_profile(data):
    """Return a real profile with typical offload and visible mechanics nonlinearity."""
    count = np.asarray(data["offload_pct"]).shape[0]
    if not 1 <= SELECTED_PROFILE_NUMBER <= count:
        raise ValueError(f"Profile {SELECTED_PROFILE_NUMBER} is outside the {count}-profile test set")
    return SELECTED_PROFILE_NUMBER - 1


def style_plot(axis, xlabel, ylabel=None):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7d8b94")
    axis.grid(True, color=GRID, linewidth=0.65, alpha=0.8)
    axis.tick_params(labelsize=7, colors=DARK, length=2.5)
    axis.set_xlabel(xlabel, fontsize=8, color=DARK, labelpad=2)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=8, color=DARK, labelpad=2)


def draw_network(axis):
    """Draw a compact spring-and-node symbol without limbs."""
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-.62, .72)
    axis.set_aspect("equal")
    axis.axis("off")
    nodes = np.array([
        [-.88, .18], [-.67, -.46], [-.48, .47], [-.31, -.02],
        [-.02, .34], [.14, -.29], [.43, -.04], [.69, .46], [.88, -.13],
    ])
    connections = [(0, 2), (0, 3), (0, 5), (1, 3), (1, 5), (2, 3),
                   (2, 4), (3, 4), (3, 6), (3, 8), (4, 6), (4, 7),
                   (5, 6), (5, 8), (6, 7), (6, 8)]
    for node_a, node_b in connections:
        a, b = nodes[node_a], nodes[node_b]
        axis.plot([a[0], b[0]], [a[1], b[1]], color=ORANGE,
                  linewidth=1.05, alpha=.78, zorder=1)
    axis.scatter(nodes[[0, 1, 2], 0], nodes[[0, 1, 2], 1], s=24, color=BLUE,
                 edgecolor="white", linewidth=.4, zorder=3)
    axis.scatter(nodes[[7, 8], 0], nodes[[7, 8], 1], s=24, color=GREEN,
                 edgecolor="white", linewidth=.4, zorder=3)
    axis.scatter(nodes[[3, 4, 5, 6], 0], nodes[[3, 4, 5, 6], 1], s=28, color="#f4a261",
                 edgecolor="white", linewidth=.4, zorder=3)


def add_arrow(figure, start, stop):
    arrow = FancyArrowPatch(start, stop, transform=figure.transFigure,
                            arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.4, color=DARK, clip_on=False)
    figure.add_artist(arrow)


def add_group_box(figure, bounds, edgecolor="#7d8b94", facecolor="white", linewidth=1.2):
    """Add a box behind a group of axes using figure coordinates."""
    x, y, width, height = bounds
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=.006,rounding_size=.008",
        transform=figure.transFigure, facecolor=facecolor,
        edgecolor=edgecolor, linewidth=linewidth, zorder=-5,
    )
    figure.add_artist(patch)


def save_asset(figure, stem, transparent=True):
    """Save one slide-ready component as vector SVG and high-resolution PNG."""
    for suffix in (".svg", ".png"):
        figure.savefig(
            ASSET_DIR / f"{stem}{suffix}", dpi=300, bbox_inches="tight",
            pad_inches=.04, transparent=transparent,
        )
    plt.close(figure)


def export_powerpoint_assets(time, angle_deg, target, spring):
    """Export individual visuals for manual assembly in PowerPoint or Slides."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    input_figure, input_axis = plt.subplots(figsize=(5.0, 2.8))
    input_axis.plot(time, target, color=BLUE, linewidth=2.4)
    style_plot(input_axis, "Time (s)", "Torque (N m)")
    input_axis.set_title("Target torque vs. time", fontsize=11, weight="bold", color=DARK, pad=6)
    input_figure.tight_layout()
    save_asset(input_figure, "01_input_target_torque_time")

    mlp_figure, mlp_axis = plt.subplots(figsize=(4.6, 2.4))
    mlp_axis.set_xlim(0, 1)
    mlp_axis.set_ylim(0, 1)
    mlp_axis.axis("off")
    layer_x = [.18, .50, .82]
    layer_y = [np.linspace(.18, .82, 5), np.linspace(.12, .88, 7), np.linspace(.24, .76, 4)]
    for source_x, source_y, target_x, target_y in zip(
        layer_x[:-1], layer_y[:-1], layer_x[1:], layer_y[1:]
    ):
        for y0 in source_y:
            for y1 in target_y:
                mlp_axis.plot([source_x, target_x], [y0, y1], color="#b8aecf",
                              linewidth=.8, alpha=.65, zorder=1)
    for x, ys, color in zip(layer_x, layer_y, [BLUE, "#6f5aa8", GREEN]):
        mlp_axis.scatter(np.full_like(ys, x), ys, s=46, color=color,
                         edgecolor="white", linewidth=.6, zorder=2)
    save_asset(mlp_figure, "02_stiffness_controller_mlp")

    network_figure, network_axis = plt.subplots(figsize=(4.6, 2.6))
    draw_network(network_axis)
    save_asset(network_figure, "03_spring_network")

    output_figure, output_axis = plt.subplots(figsize=(5.0, 3.2))
    output_axis.plot(time, target, color=BLUE, linewidth=2.2, label="Target torque")
    output_axis.plot(time, spring, color=ORANGE, linewidth=2.2, label="Spring-network torque")
    sample_marks = np.arange(0, time.size, 8)
    output_axis.scatter(time[sample_marks], spring[sample_marks], s=14, color=ORANGE,
                        edgecolor="white", linewidth=.35, zorder=4)
    style_plot(output_axis, "Time (s)", "Torque (N m)")
    output_axis.set_title("Torque vs. time", fontsize=11, weight="bold", color=DARK, pad=6)
    output_axis.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
                       bbox_to_anchor=(.5, -.18))
    output_figure.subplots_adjust(left=.14, right=.98, top=.86, bottom=.28)
    save_asset(output_figure, "04_output_torque_time")

    angle_figure, angle_axis = plt.subplots(figsize=(5.0, 3.2))
    angle_axis.plot(angle_deg, target, color=BLUE, linewidth=2.2)
    style_plot(angle_axis, "Joint angle (deg)", "Torque (N m)")
    angle_axis.set_title("Torque vs. joint angle", fontsize=11, weight="bold", color=DARK, pad=6)
    angle_figure.subplots_adjust(left=.14, right=.98, top=.86, bottom=.18)
    save_asset(angle_figure, "06_torque_angle_profile")

    time_figure, time_axis = plt.subplots(figsize=(5.0, 3.2))
    time_axis.plot(time, target, color=BLUE, linewidth=2.2)
    style_plot(time_axis, "Time (s)", "Torque (N m)")
    time_axis.set_title("Torque vs. time", fontsize=11, weight="bold", color=DARK, pad=6)
    time_figure.subplots_adjust(left=.14, right=.98, top=.86, bottom=.18)
    save_asset(time_figure, "07_torque_time_profile")

    paired_figure, (paired_angle, paired_time) = plt.subplots(1, 2, figsize=(9.4, 3.4))
    paired_angle.plot(angle_deg, target, color=BLUE, linewidth=2.2)
    paired_time.plot(time, target, color=BLUE, linewidth=2.2)
    style_plot(paired_angle, "Joint angle (deg)", "Torque (N m)")
    style_plot(paired_time, "Time (s)", "Torque (N m)")
    paired_angle.set_title("Torque vs. joint angle", fontsize=11, weight="bold",
                           color=DARK, pad=6)
    paired_time.set_title("Torque vs. time", fontsize=11, weight="bold",
                          color=DARK, pad=6)
    paired_angle.set_ylim(-22, 104)
    paired_time.set_ylim(-22, 104)
    add_group_box(paired_figure, [.015, .035, .475, .93], edgecolor=BLUE,
                  facecolor=PALE, linewidth=1.2)
    add_group_box(paired_figure, [.510, .035, .475, .93], edgecolor=BLUE,
                  facecolor=PALE, linewidth=1.2)
    paired_angle.set_facecolor("none")
    paired_time.set_facecolor("none")
    paired_figure.subplots_adjust(left=.065, right=.965, top=.85, bottom=.18, wspace=.30)
    save_asset(paired_figure, "08_target_torque_angle_time_profiles")

    topology = plt.imread(FIGURE_ONE_PATH)[90:, ...].copy()
    plt.imsave(ASSET_DIR / "05_spring_network_topology.png", topology)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with np.load(DATA_PATH) as data:
        index = choose_representative_profile(data)
        time = np.asarray(data["time"])[index]
        angle_deg = np.rad2deg(np.asarray(data["theta"])[index])
        target = np.asarray(data["target_torque"])[index]
        spring = np.asarray(data["spring_torque"])[index, -1]
        offload = float(np.asarray(data["offload_pct"])[index, -1])
        rmse = float(np.asarray(data["rmse"])[index, -1])

    export_powerpoint_assets(time, angle_deg, target, spring)

    figure = plt.figure(figsize=(14.8, 9.2), dpi=220, facecolor="white")

    # Left: top-down flowchart. Right: the established Figure 1 rendering.
    left_x, left_w = .038, .280
    right_x, right_w = .337, .658

    # Stage 1: the target trajectory is the input to the learned controller.
    add_group_box(figure, [left_x, .650, left_w, .235], edgecolor=BLUE, facecolor=PALE)
    figure.text(left_x + .020, .855, "1  INPUT", ha="left",
                fontsize=9.2, weight="bold", color=DARK)
    before_time = figure.add_axes([left_x + .054, .690, .172, .120])
    before_time.plot(time, target, color=BLUE, linewidth=1.8)
    style_plot(before_time, "Time (s)", "Torque (N m)")
    before_time.set_title("Target torque vs. time", fontsize=8.5, weight="bold", color=DARK, pad=4)

    # Stage 2: learned controller maps the observed profile to spring stiffnesses.
    add_group_box(figure, [left_x, .490, left_w, .125], edgecolor=BLUE, facecolor=PALE)
    figure.text(left_x + .020, .588, "2  STIFFNESS CONTROLLER MLP", ha="left",
                fontsize=9.2, weight="bold", color=DARK)
    controller_axis = figure.add_axes([left_x + .035, .497, .210, .088])
    controller_axis.axis("off")
    layer_x = [.24, .50, .76]
    layer_y = [np.linspace(.28, .78, 4), np.linspace(.25, .81, 5), np.linspace(.32, .74, 3)]
    for source_x, source_y, target_x, target_y in zip(
        layer_x[:-1], layer_y[:-1], layer_x[1:], layer_y[1:]
    ):
        for y0 in source_y:
            for y1 in target_y:
                controller_axis.plot([source_x, target_x], [y0, y1], color="#b8aecf",
                                     linewidth=.45, alpha=.65,
                                     transform=controller_axis.transAxes, zorder=1)
    for x, ys, color in zip(layer_x, layer_y, [BLUE, "#6f5aa8", GREEN]):
        controller_axis.scatter(np.full_like(ys, x), ys, s=18, color=color,
                                edgecolor="white", linewidth=.35,
                                transform=controller_axis.transAxes, zorder=2)
    # Stage 3: compact network stage.
    add_group_box(figure, [left_x, .320, left_w, .145], edgecolor=BLUE, facecolor=PALE)
    figure.text(left_x + .020, .438, "3  SPRING NETWORK", ha="left",
                fontsize=9.2, weight="bold", color=DARK)
    network_axis = figure.add_axes([left_x + .060, .327, left_w - .12, .095])
    draw_network(network_axis)

    # Box 4: same profile and samples, now with the mechanics torque overlaid over time.
    add_group_box(figure, [left_x, .025, left_w, .245], edgecolor=BLUE, facecolor=PALE)
    figure.text(left_x + .020, .242, "4  OUTPUT",
                ha="left", fontsize=9.2, weight="bold", color=DARK)
    after_time = figure.add_axes([left_x + .054, .085, .172, .105])
    after_time.plot(time, target, color=BLUE, linewidth=1.55, label="Target torque")
    after_time.plot(time, spring, color=ORANGE, linewidth=1.55, label="Spring-network torque")
    sample_marks = np.arange(0, time.size, 8)
    after_time.scatter(time[sample_marks], spring[sample_marks], s=8, color=ORANGE,
                       edgecolor="white", linewidth=.25, zorder=4)
    style_plot(after_time, "Time (s)", "Torque (N m)")
    after_time.set_title("Torque vs. time", fontsize=8.5, weight="bold", color=DARK, pad=4)
    handles, labels = after_time.get_legend_handles_labels()
    figure.legend(handles, labels, loc="center", bbox_to_anchor=(left_x + left_w / 2, .213), ncol=2,
                  frameon=False, fontsize=7.8, handlelength=2.2)
    # Existing Figure 1, retained intact on the right.
    fig1_axis = figure.add_axes([right_x, .035, right_w, .845])
    fig1_image = plt.imread(FIGURE_ONE_PATH)[90:, ...].copy()
    fig1_axis.imshow(fig1_image)
    fig1_axis.axis("off")
    figure.text(right_x + right_w / 2, .965, "Spring network topology",
                ha="center", va="top", fontsize=15, weight="bold", color=DARK)
    figure.suptitle("Adaptive spring network pipeline", x=.045, ha="left",
                    y=.965, fontsize=15, weight="bold", color=DARK)

    # Vertical arrows connect every boxed stage in the left flowchart.
    center_x = left_x + left_w / 2
    add_arrow(figure, (center_x, .642), (center_x, .622))
    add_arrow(figure, (center_x, .482), (center_x, .474))
    add_arrow(figure, (center_x, .312), (center_x, .278))

    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    print(f"Selected profile: {index + 1}")
    print(f"Settled RMSE: {rmse:.6f} N m")
    print(f"Settled offload: {offload:.6f}%")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
