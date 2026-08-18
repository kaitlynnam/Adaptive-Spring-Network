"""Generate the paper diagram for the causal one-period-buffer controller."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "plots" / "period_adaptive_3d" / "fig02_causal_period_pipeline.png"


def add_box(axis, center, size, text, facecolor, edgecolor):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=2, facecolor=facecolor, edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9.5,
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
    fig, axis = plt.subplots(figsize=(16, 6.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(0.5, 0.94, "Causal One-Period-Buffer 60-Spring Pipeline",
              ha="center", va="center", fontsize=21)

    centers = [(0.09, 0.61), (0.295, 0.61), (0.50, 0.61),
               (0.705, 0.61), (0.91, 0.61)]
    widths = [0.15, 0.15, 0.15, 0.16, 0.15]
    labels = [
        "Completed Period n\n6 channels × 160 samples",
        "Period-conditioned MLP\n960 → 256 → 60",
        "60 positive bounded\nstiffnesses",
        "Hold throughout Period n + 1\nrelaxed 3D mechanics",
        "Spring torque +\nresidual motor torque",
    ]
    for center, width, label in zip(centers, widths, labels):
        add_box(axis, center, (width, 0.13), label, "#e7f0f7", "#35627d")
    for index in range(len(centers) - 1):
        arrow(axis, (centers[index][0] + widths[index] / 2 + 0.008, 0.61),
              (centers[index + 1][0] - widths[index + 1] / 2 - 0.008, 0.61))

    loss_center = (0.68, 0.26)
    add_box(axis, loss_center, (0.34, 0.11),
            "Training loss\nnext-period torque MSE + stiffness-change penalty",
            "#fbe7db", "#bc5427")
    axis.annotate(
        "", xy=(0.78, 0.32), xytext=(0.91, 0.535),
        arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#b95025"},
    )
    axis.text(0.90, 0.43, "compare with target", ha="center", fontsize=9)
    axis.annotate(
        "", xy=(0.295, 0.535), xytext=(0.53, 0.26),
        arrowprops={"arrowstyle": "-|>", "linewidth": 2, "color": "#b95025",
                    "connectionstyle": "arc3,rad=-0.28"},
    )
    axis.text(0.43, 0.38, "Backpropagation\nupdates MLP weights",
              ha="center", va="center", fontsize=9, color="#a54822")

    axis.text(0.5, 0.08,
              "Period 1 uses default stiffness with no neural input; every prediction is applied one period later.",
              ha="center", fontsize=10, color="#444444")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
