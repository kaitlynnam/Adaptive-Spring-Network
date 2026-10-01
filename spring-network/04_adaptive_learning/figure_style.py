"""Shared extended-abstract export style for publication figures."""

SPRING_COLOR = "#2f6f9f"
MOTOR_COLOR = "#d97720"


def save_publication_figure(figure, path, dpi=220):
    """Export transparent PNG and editable SVG without redundant plot titles."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for axis in figure.axes:
        axis.set_title("")
    for destination in (path, path.with_suffix(".svg")):
        figure.savefig(destination, dpi=dpi, transparent=True, bbox_inches="tight")
