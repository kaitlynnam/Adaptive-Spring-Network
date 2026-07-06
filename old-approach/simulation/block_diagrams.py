from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class Block:
    id: str
    label: str
    x: float
    y: float
    width: float = 180.0
    height: float = 64.0


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    label: str = ""


@dataclass(frozen=True)
class Diagram:
    title: str
    blocks: list[Block]
    connections: list[Connection]
    width: int = 980
    height: int = 620

    def to_svg(self) -> str:
        block_by_id = {block.id: block for block in self.blocks}
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#1f2933" />',
            "</marker>",
            "</defs>",
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{self.width / 2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{escape(self.title)}</text>',
        ]
        for connection in self.connections:
            parts.append(_connection_svg(block_by_id[connection.source], block_by_id[connection.target], connection.label))
        for block in self.blocks:
            parts.append(_block_svg(block))
        parts.append("</svg>")
        return "\n".join(parts)

    def write_svg(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_svg())
        return output_path


def adaptive_pej_loop_diagram() -> Diagram:
    return Diagram(
        title="Adaptive PEJ Simulation Loop",
        blocks=[
            Block("terrain", "Terrain / q schedule", 60, 90),
            Block("controller", "Controller\nrequired joint torque", 300, 90),
            Block("spring", "Spring model\nfixed / blend / actuator", 300, 240),
            Block("motor", "Motor model\nresidual torque + power", 540, 170),
            Block("dynamics", "Joint dynamics\nsolve_ivp", 740, 170),
            Block("state", "Joint state\ntheta, theta_dot", 540, 360),
            Block("energy", "Energy analysis\noffload metrics", 300, 430),
            Block("plots", "Plots / notebook\ncomparison figures", 60, 430),
        ],
        connections=[
            Connection("terrain", "spring", "q"),
            Connection("controller", "motor", "tau_required"),
            Connection("spring", "motor", "tau_spring"),
            Connection("motor", "dynamics", "tau_motor + tau_spring"),
            Connection("dynamics", "state", "integrated state"),
            Connection("state", "spring", "theta"),
            Connection("state", "controller", "theta, theta_dot"),
            Connection("motor", "energy", "power"),
            Connection("state", "energy", "time histories"),
            Connection("energy", "plots", "summaries"),
        ],
    )


def spring_model_comparison_diagram() -> Diagram:
    return Diagram(
        title="Interchangeable Spring Models",
        blocks=[
            Block("input", "Inputs\ntheta, q", 60, 240),
            Block("fixed", "FixedSpringModel\ntau = k theta", 300, 90),
            Block("blend", "AdaptiveBlendModel\nblend flat/rough profiles", 300, 240),
            Block("actuator", "ActuatorTunedModel\nq -> k_eff -> torque", 300, 390),
            Block("output", "Common output\ntau_spring", 620, 240),
        ],
        connections=[
            Connection("input", "fixed", "theta"),
            Connection("input", "blend", "theta, q"),
            Connection("input", "actuator", "theta, q"),
            Connection("fixed", "output", "tau"),
            Connection("blend", "output", "tau"),
            Connection("actuator", "output", "tau"),
        ],
        height=560,
    )


def actuator_tuned_diagram() -> Diagram:
    return Diagram(
        title="Actuator-Tuned Stiffness Model",
        blocks=[
            Block("roughness", "Encoder motion\nroughness estimate", 60, 100),
            Block("q", "Clamp q\n0 to 1", 300, 100),
            Block("phi", "Tuning actuator\nphi command / lag", 540, 100),
            Block("stiffness", "Effective stiffness\nk_eff", 540, 270),
            Block("theta", "Joint angle\ntheta", 60, 350),
            Block("torque", "Spring torque\ntau = k_eff theta", 300, 350),
            Block("motor", "Motor residual\ntau_motor = tau_required - tau", 620, 430, width=230),
        ],
        connections=[
            Connection("roughness", "q", "score"),
            Connection("q", "phi", "q"),
            Connection("q", "stiffness", "q"),
            Connection("phi", "stiffness", "optional physical state"),
            Connection("stiffness", "torque", "k_eff"),
            Connection("theta", "torque", "theta"),
            Connection("torque", "motor", "tau_spring"),
        ],
        height=600,
    )


def cam_spring_network_diagram() -> Diagram:
    return Diagram(
        title="Cam-Controlled 3-Spring Adaptive PEJ Network",
        blocks=[
            Block("terrain", "Terrain / Roughness\nq", 60, 80),
            Block("phi_des", "Desired Cam Angle\nphi_des", 300, 80),
            Block("actuator", "Cam Actuator Dynamics\nspeed limit + lag", 540, 80),
            Block("phi", "Actual Cam Angle\nphi", 780, 80),
            Block("geometry", "Cam Geometry\nengagement ramps", 300, 245),
            Block("springs", "3-Spring Network\ncompressions + forces", 540, 245),
            Block("torque", "Passive Spring Torque", 780, 245),
            Block("dynamics", "Joint Dynamics", 540, 410),
            Block("power", "Motor Power + Cam Power", 300, 410),
            Block("energy", "Net Energy Savings", 60, 410),
        ],
        connections=[
            Connection("terrain", "phi_des", "q"),
            Connection("phi_des", "actuator", "phi_des"),
            Connection("actuator", "phi", "phi"),
            Connection("phi", "geometry", "cam angle"),
            Connection("geometry", "springs", "engagement"),
            Connection("springs", "torque", "forces"),
            Connection("torque", "dynamics", "tau_spring"),
            Connection("dynamics", "power", "theta_dot"),
            Connection("actuator", "power", "cam power"),
            Connection("power", "energy", "E_motor - E_cam"),
        ],
        height=560,
    )


def repo_workflow_diagram() -> Diagram:
    return Diagram(
        title="Adaptive PEJ Repo Workflow",
        blocks=[
            Block("inputs", "Trajectory Data\nCSV / NPZ / synthetic", 60, 90),
            Block("pej_math", "PEJ Math Package\npower, offload, distillation", 300, 90),
            Block("models", "Spring Models\nfixed / adaptive / cam", 540, 90),
            Block("simulation", "Simulation Framework\nSciPy + block wiring", 300, 250),
            Block("energy", "Energy Analysis\nmotor + cam cost", 540, 250),
            Block("examples", "Examples + Notebook\nrun studies", 60, 410),
            Block("outputs", "Artifacts\nplots, tables, diagrams", 540, 410),
        ],
        connections=[
            Connection("inputs", "pej_math", "theta, theta_dot, tau_total"),
            Connection("pej_math", "models", "profiles + torque helpers"),
            Connection("models", "simulation", "tau_spring"),
            Connection("simulation", "energy", "time histories"),
            Connection("energy", "outputs", "metrics"),
            Connection("examples", "inputs", "load/generate"),
            Connection("examples", "simulation", "configure/run"),
            Connection("examples", "outputs", "save"),
        ],
        height=540,
    )


def write_default_diagrams(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    diagrams = {
        "repo_workflow.svg": repo_workflow_diagram(),
        "adaptive_pej_loop.svg": adaptive_pej_loop_diagram(),
        "spring_model_comparison.svg": spring_model_comparison_diagram(),
        "actuator_tuned_stiffness.svg": actuator_tuned_diagram(),
        "cam_spring_network.svg": cam_spring_network_diagram(),
    }
    return [diagram.write_svg(output_path / filename) for filename, diagram in diagrams.items()]


def _block_svg(block: Block) -> str:
    label_lines = block.label.splitlines()
    line_height = 18
    first_y = block.y + block.height / 2 - (len(label_lines) - 1) * line_height / 2 + 5
    text = "\n".join(
        f'<text x="{block.x + block.width / 2}" y="{first_y + i * line_height}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{escape(line)}</text>'
        for i, line in enumerate(label_lines)
    )
    return "\n".join(
        [
            f'<rect x="{block.x}" y="{block.y}" width="{block.width}" height="{block.height}" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.5" />',
            text,
        ]
    )


def _connection_svg(source: Block, target: Block, label: str) -> str:
    x1, y1, x2, y2 = _edge_points(source, target)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2 - 8
    label_svg = ""
    if label:
        label_svg = (
            f'<rect x="{mid_x - 45}" y="{mid_y - 15}" width="90" height="20" rx="4" fill="#ffffff" opacity="0.9" />'
            f'<text x="{mid_x}" y="{mid_y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">{escape(label)}</text>'
        )
    return "\n".join(
        [
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1f2933" stroke-width="1.4" marker-end="url(#arrow)" />',
            label_svg,
        ]
    )


def _edge_points(source: Block, target: Block) -> tuple[float, float, float, float]:
    source_cx = source.x + source.width / 2
    source_cy = source.y + source.height / 2
    target_cx = target.x + target.width / 2
    target_cy = target.y + target.height / 2
    dx = target_cx - source_cx
    dy = target_cy - source_cy
    if abs(dx) >= abs(dy):
        x1 = source.x + source.width if dx >= 0 else source.x
        y1 = source_cy
        x2 = target.x if dx >= 0 else target.x + target.width
        y2 = target_cy
    else:
        x1 = source_cx
        y1 = source.y + source.height if dy >= 0 else source.y
        x2 = target_cx
        y2 = target.y if dy >= 0 else target.y + target.height
    return x1, y1, x2, y2
