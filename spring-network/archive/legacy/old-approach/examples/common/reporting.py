from __future__ import annotations

import csv
from pathlib import Path


STANDARD_COLUMNS = [
    "scenario",
    "case",
    "baseline_motor_energy_j",
    "motor_energy_j",
    "mean_power_w",
    "cam_energy_j",
    "net_saved_j",
    "offload_pct",
    "mean_q",
    "mean_phi",
    "mean_k",
    "spring_k",
    "spring_k_1",
    "spring_k_2",
    "spring_k_3",
    "peak_motor_power_w",
    "constraints_passed",
]


def print_section(title: str) -> None:
    print(f"\n{title}")


def print_table(rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    if not rows:
        return
    selected = columns or [column for column in STANDARD_COLUMNS if any(row.get(column, "") != "" for row in rows)]
    formatted = [[_format_cell(row.get(column, "")) for column in selected] for row in rows]
    widths = [len(column) for column in selected]
    for row in formatted:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*selected))
    print(fmt.format(*["-" * width for width in widths]))
    for row in formatted:
        print(fmt.format(*row))


def write_csv(path: str | Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = columns or [column for column in STANDARD_COLUMNS if any(row.get(column, "") != "" for row in rows)]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column, "")) for column in selected})
    return output_path


def print_written(paths: list[Path] | tuple[Path, ...], label: str = "Wrote") -> None:
    for path in paths:
        print(f"  {label}: {path}")


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
