"""Generate the paper diagram for the causal one-period-buffer controller."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "plots" / "period_adaptive_3d" / "fig02_causal_period_pipeline.png"


def add_box(axis, center, size, text, facecolor, edgecolor, fontsize=9.5):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=2, facecolor=facecolor, edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize,
              color="#20252b", linespacing=1.35)


def arrow(axis, start, stop, label=None, curve=0.0):
    axis.annotate(
        "", xy=stop, xytext=start,
        arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#4a5560",
                    "connectionstyle": f"arc3,rad={curve}"},
    )
    if label:
        axis.text((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2 + 0.035,
                  label, ha="center", va="bottom", fontsize=9, color="#4a5560")


def main():
    fig, axis = plt.subplots(figsize=(18, 6.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.5, 0.94, "From Period Measurements to the Training Loss",
              ha="center", va="center", fontsize=21, weight="bold")
    axis.text(
        0.5, 0.885,
        "The controller observes one completed period and selects the stiffness held during the next period.",
        ha="center", va="center", fontsize=11, color="#4a5560",
    )

    centers = [(0.09, 0.62), (0.295, 0.62), (0.50, 0.62),
               (0.705, 0.62), (0.91, 0.62)]
    widths = [0.15] * 5
    labels = [
        "1  NORMALIZED INPUT\nCompleted Period n\nθ, θ̇, θ̈, target,\nspring and motor torque\n6 × 160 = 960 values",
        "2  STIFFNESS CONTROLLER\n960 inputs\n256 tanh units\n60 outputs",
        "3  STIFFNESS COMMAND\n60 bounded values",
        "4  MECHANICS MODEL\nDifferentiable surrogate\nor full mechanics solver",
        "5  TORQUE ERROR\n$\\tau_{motor}(t)=\\tau_{target}(t)$\n$-\\tau_{spring}(t)$",
    ]
    colors = [
        ("#e8f1f8", "#35627d"), ("#e9e5f5", "#62528f"),
        ("#e8f4ec", "#39825c"), ("#fff2d9", "#b27a18"),
        ("#fde9e7", "#b84f47"),
    ]
    for center, width, label, (facecolor, edgecolor) in zip(
        centers, widths, labels, colors
    ):
        add_box(axis, center, (width, 0.20), label, facecolor, edgecolor, fontsize=9.2)
    for index in range(len(centers) - 1):
        arrow(axis, (centers[index][0] + widths[index] / 2 + 0.006, 0.62),
              (centers[index + 1][0] - widths[index + 1] / 2 - 0.006, 0.62))

    loss_center = (0.65, 0.29)
    add_box(
        axis, loss_center, (0.25, 0.105),
        "6  TRAINING LOSS\n$\\mathcal{L}=\\mathrm{MSE}(\\tau_{motor})+0.1\\,\\mathcal{L}_{stiffness}$",
        "#fbe3d5", "#b95025", fontsize=9.4,
    )
    axis.annotate(
        "", xy=(0.73, 0.355), xytext=(0.88, 0.51),
        arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#b95025"},
    )

    # The loss trains the MLP; this is the only backward-flow arrow.
    axis.annotate(
        "", xy=(0.295, 0.51), xytext=(0.53, 0.29),
        arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#b95025",
                    "connectionstyle": "arc3,rad=-0.28"},
    )
    axis.text(0.445, 0.365, "Backpropagation\nupdates MLP weights",
              ha="center", va="center", fontsize=9.2, color="#a54822")

    axis.text(
        0.5, 0.105,
        "Causal deployment: Period 1 uses the topology's default stiffness. Thereafter, each completed period determines the stiffness for the following period.",
        ha="center", fontsize=10, color="#444444",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f5f6f7",
              "edgecolor": "#c7ccd1"},
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
