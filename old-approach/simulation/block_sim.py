from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .dynamics import JointSimulationConfig, JointSimulationResult, simulate_joint
from .spring_models import ActuatorTunedModel


SignalFunction = Callable[[float, dict[str, float]], dict[str, float]]


@dataclass(frozen=True)
class PortRef:
    block: str
    port: str

    @classmethod
    def parse(cls, value: str) -> "PortRef":
        block, port = value.split(".", maxsplit=1)
        return cls(block=block, port=port)


@dataclass(frozen=True)
class SignalConnection:
    source: PortRef
    target: PortRef


@dataclass(frozen=True)
class ComputationBlock:
    """Small executable block with named inputs and outputs."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    compute: SignalFunction


@dataclass
class ExecutableBlockDiagram:
    """Minimal Simulink-like block diagram for algebraic signal wiring."""

    blocks: list[ComputationBlock] = field(default_factory=list)
    connections: list[SignalConnection] = field(default_factory=list)

    def add_block(self, block: ComputationBlock) -> None:
        if any(existing.name == block.name for existing in self.blocks):
            raise ValueError(f"block {block.name!r} already exists")
        self.blocks.append(block)

    def connect(self, source: str, target: str) -> None:
        self.connections.append(SignalConnection(PortRef.parse(source), PortRef.parse(target)))

    def evaluate(self, t: float, external_signals: dict[str, float]) -> dict[str, float]:
        """Evaluate all blocks once at time t using named signal connections."""

        signal_values = dict(external_signals)
        pending = list(self.blocks)
        while pending:
            next_pending = []
            progressed = False
            for block in pending:
                inputs = self._inputs_for_block(block, signal_values)
                if inputs is None:
                    next_pending.append(block)
                    continue
                outputs = block.compute(t, inputs)
                missing_outputs = [name for name in block.outputs if name not in outputs]
                if missing_outputs:
                    raise ValueError(f"block {block.name!r} did not produce outputs: {missing_outputs}")
                for output_name in block.outputs:
                    signal_values[f"{block.name}.{output_name}"] = float(outputs[output_name])
                progressed = True
            if not progressed:
                unresolved = ", ".join(block.name for block in next_pending)
                raise RuntimeError(f"could not resolve block inputs for: {unresolved}")
            pending = next_pending
        return signal_values

    def _inputs_for_block(
        self,
        block: ComputationBlock,
        signal_values: dict[str, float],
    ) -> dict[str, float] | None:
        values = {}
        for input_name in block.inputs:
            source = self._source_for_input(block.name, input_name)
            signal_name = f"{source.block}.{source.port}" if source is not None else input_name
            if signal_name not in signal_values:
                return None
            values[input_name] = signal_values[signal_name]
        return values

    def _source_for_input(self, block_name: str, input_name: str) -> PortRef | None:
        matches = [
            connection.source
            for connection in self.connections
            if connection.target.block == block_name and connection.target.port == input_name
        ]
        if len(matches) > 1:
            raise ValueError(f"multiple sources connected to {block_name}.{input_name}")
        return matches[0] if matches else None


@dataclass(frozen=True)
class PEJBlockSimulation:
    """Executable adaptive PEJ block diagram backed by the SciPy ODE solver."""

    diagram: ExecutableBlockDiagram
    spring_model: object
    controller: object
    q_input: Callable[[float], float]
    config: JointSimulationConfig

    def simulate(self) -> JointSimulationResult:
        return simulate_joint(
            spring_model=self.spring_model,
            controller=self.controller,
            q_input=self.q_input,
            config=self.config,
        )

    def sample_signals(self, result: JointSimulationResult, index: int) -> dict[str, float]:
        """Evaluate the same block wiring at one simulated sample."""

        theta = float(result.theta[index])
        theta_dot = float(result.theta_dot[index])
        q = float(result.q[index])
        phi = None if np.isnan(result.phi[index]) else float(result.phi[index])
        tau_required = float(result.tau_required[index])
        external = {
            "joint_state.theta": theta,
            "joint_state.theta_dot": theta_dot,
            "q_source.q": q,
            "controller.tau_required": tau_required,
        }
        if phi is not None:
            external["actuator.phi"] = phi
        return self.diagram.evaluate(float(result.time[index]), external)


def build_pej_block_simulation(
    *,
    spring_model,
    controller,
    q_input: Callable[[float], float],
    config: JointSimulationConfig,
) -> PEJBlockSimulation:
    """Build an executable block diagram for the adaptive PEJ simulation."""

    diagram = ExecutableBlockDiagram()
    spring_inputs = ("theta", "q", "phi") if isinstance(spring_model, ActuatorTunedModel) else ("theta", "q")
    diagram.add_block(
        ComputationBlock(
            name="spring_model",
            inputs=spring_inputs,
            outputs=("tau_spring",),
            compute=lambda t, u: {
                "tau_spring": spring_model.compute_torque(
                    u["theta"],
                    u["q"],
                    phi=u.get("phi"),
                )
                if isinstance(spring_model, ActuatorTunedModel)
                else spring_model.compute_torque(u["theta"], u["q"])
            },
        )
    )
    diagram.add_block(
        ComputationBlock(
            name="motor_model",
            inputs=("tau_required", "tau_spring", "theta_dot"),
            outputs=("tau_motor", "motor_power"),
            compute=lambda t, u: {
                "tau_motor": u["tau_required"] - u["tau_spring"],
                "motor_power": max((u["tau_required"] - u["tau_spring"]) * u["theta_dot"], 0.0),
            },
        )
    )
    diagram.add_block(
        ComputationBlock(
            name="joint_dynamics",
            inputs=("tau_motor", "tau_spring", "theta_dot"),
            outputs=("theta_ddot",),
            compute=lambda t, u: {
                "theta_ddot": (u["tau_motor"] + u["tau_spring"] - config.damping * u["theta_dot"])
                / config.inertia
            },
        )
    )
    diagram.connect("joint_state.theta", "spring_model.theta")
    diagram.connect("q_source.q", "spring_model.q")
    if isinstance(spring_model, ActuatorTunedModel):
        diagram.connect("actuator.phi", "spring_model.phi")
    diagram.connect("controller.tau_required", "motor_model.tau_required")
    diagram.connect("spring_model.tau_spring", "motor_model.tau_spring")
    diagram.connect("joint_state.theta_dot", "motor_model.theta_dot")
    diagram.connect("motor_model.tau_motor", "joint_dynamics.tau_motor")
    diagram.connect("spring_model.tau_spring", "joint_dynamics.tau_spring")
    diagram.connect("joint_state.theta_dot", "joint_dynamics.theta_dot")
    return PEJBlockSimulation(
        diagram=diagram,
        spring_model=spring_model,
        controller=controller,
        q_input=q_input,
        config=config,
    )
