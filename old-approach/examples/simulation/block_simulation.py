#!/usr/bin/env python3
"""Run the adaptive PEJ as an executable Python block diagram."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from simulation.block_sim import build_pej_block_simulation
from simulation.controller import FeedforwardTorqueController
from simulation.dynamics import JointSimulationConfig
from simulation.spring_models import ActuatorTunedModel
from simulation.terrain import StepTerrainSchedule


def main() -> None:
    terrain = StepTerrainSchedule(switch_time=1.0, q_before=0.0, q_after=1.0)
    block_sim = build_pej_block_simulation(
        spring_model=ActuatorTunedModel(
            k_soft=2.0,
            k_stiff=6.0,
            phi_min=-0.25,
            phi_max=0.25,
            actuator_time_constant=0.15,
        ),
        controller=FeedforwardTorqueController(lambda t: 0.8),
        q_input=terrain.q,
        config=JointSimulationConfig(
            inertia=0.2,
            damping=0.02,
            t_span=(0.0, 2.0),
            dt=0.02,
            initial_theta=0.1,
            initial_phi=-0.25,
        ),
    )
    result = block_sim.simulate()
    signals = block_sim.sample_signals(result, index=result.time.size // 2)

    print("Executable adaptive PEJ block simulation")
    print("  Blocks:")
    for block in block_sim.diagram.blocks:
        print(f"    {block.name}: inputs={block.inputs}, outputs={block.outputs}")
    print("  Connections:")
    for connection in block_sim.diagram.connections:
        print(
            f"    {connection.source.block}.{connection.source.port}"
            f" -> {connection.target.block}.{connection.target.port}"
        )
    print("  Mid-simulation sample:")
    for name in sorted(signals):
        print(f"    {name}: {signals[name]:.4f}")


if __name__ == "__main__":
    main()
