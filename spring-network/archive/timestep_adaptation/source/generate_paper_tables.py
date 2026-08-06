"""Create paper-ready CSV and PNG tables from preserved result artifacts."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tables"
OUTPUT = ROOT / "plots" / "paper_figures" / "tables"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_table(stem, columns, rows, title, note):
    csv_path = OUTPUT / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    width = max(10.0, 2.0 * len(columns))
    height = 2.2 + 0.42 * len(rows)
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    table = axis.table(
        cellText=rows, colLabels=columns, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    figure.suptitle(title, fontsize=13, weight="bold", y=0.96)
    figure.text(0.5, 0.035, note, ha="center", fontsize=8.5, color="0.3")
    figure.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def primary_performance():
    rows = read_rows(
        SOURCE / "spatial"
        / "global_56s_c0131_screen300_mechanics_comparison.csv"
    )
    output = []
    labels = {
        "fixed_spatial_baseline": "Fixed stiffness",
        "adaptive_spatial": "Adaptive stiffness",
    }
    for row in rows:
        output.append([
            labels[row["model"]],
            row["profiles"],
            f'{float(row["mean_rmse_nm"]):.2f}',
            f'{float(row["median_rmse_nm"]):.2f}',
            f'{float(row["mean_offload_pct"]):.2f}',
            f'{float(row["median_offload_pct"]):.2f}',
            f'{float(row["mean_abs_residual_nm"]):.2f}',
        ])
    write_table(
        "table01_primary_performance_preliminary",
        [
            "Model", "Profiles", "Mean RMSE\n[N·m]", "Median RMSE\n[N·m]",
            "Mean offload\n[%]", "Median offload\n[%]",
            "Mean |motor|\n[N·m]",
        ],
        output,
        "Primary 3D Performance Comparison",
        "Candidate 131, 56 linear springs; 300-iteration screen; relaxed 3D mechanics.",
    )


def selected_topology_feasibility():
    path = SOURCE / "spatial" / "global_candidate_0131_56s_dense_audit.json"
    with path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    output = [
        ["Springs", "56", "Selected linear-spring topology"],
        ["Angle range", "−45° to +45°", f'{audit["angle_samples"]} audited states'],
        [
            "Minimum spring clearance",
            f'{1000 * audit["minimum_spring_to_spring_clearance_m"]:.2f} mm',
            "Requirement: 50.00 mm",
        ],
        ["Spring–spring violations", audit["spring_to_spring_clearance_violations"], "None detected"],
        ["Spring–limb intersections", audit["limb_intersections"], "None detected"],
        ["Spring–bearing intersections", audit["bearing_intersections"], "None detected"],
        ["Fixed-to-fixed springs", audit["fixed_to_fixed_springs"], "None"],
        ["Connected spring graph", "Yes" if audit["spring_graph_connected"] else "No", ""],
        ["Dense audit passed", "Yes" if audit["passed"] else "No", ""],
    ]
    write_table(
        "table02_selected_topology_feasibility",
        ["Property", "Result", "Criterion / interpretation"],
        output,
        "Selected 56-Spring Topology: Geometric Feasibility",
        "Ideal centerline audit; excludes fasteners, finite spring diameter, fatigue, and structural stress.",
    )


def mechanical_convergence():
    output = []
    for law in ("linear", "cubic"):
        rows = read_rows(
            SOURCE / "mechanics_audits"
            / f"stiffness_{law}_refreshed_relax300_mechanics_audit.csv"
        )
        for row in rows:
            output.append([
                law.capitalize(),
                row["relaxation_steps"],
                f'{float(row["mean_rmse_nm"]):.2f}',
                f'{float(row["mean_offload_pct"]):.2f}',
                f'{float(row["mean_force_residual_n"]):.3f}',
                f'{float(row["torque_rmse_vs_deepest_nm"]):.3f}',
            ])
    write_table(
        "table03_mechanical_convergence",
        [
            "Spring law", "Relaxation\nsteps", "Mean RMSE\n[N·m]",
            "Mean offload\n[%]", "Mean force\nresidual [N]",
            "Torque RMSE vs.\ndeepest [N·m]",
        ],
        output,
        "Mechanical Relaxation Convergence",
        "Thirty profiles per condition; deepest available evaluation is 500 relaxation steps.",
    )


if __name__ == "__main__":
    primary_performance()
    selected_topology_feasibility()
    mechanical_convergence()
    print(OUTPUT)
