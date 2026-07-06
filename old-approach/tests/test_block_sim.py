from simulation.block_sim import ComputationBlock, ExecutableBlockDiagram, build_pej_block_simulation
from simulation.controller import FeedforwardTorqueController
from simulation.dynamics import JointSimulationConfig
from simulation.spring_models import FixedSpringModel


def test_executable_block_diagram_propagates_connected_signals():
    diagram = ExecutableBlockDiagram()
    diagram.add_block(
        ComputationBlock(
            name="gain",
            inputs=("x",),
            outputs=("y",),
            compute=lambda t, u: {"y": 2.0 * u["x"]},
        )
    )
    diagram.add_block(
        ComputationBlock(
            name="sum",
            inputs=("a", "b"),
            outputs=("y",),
            compute=lambda t, u: {"y": u["a"] + u["b"]},
        )
    )
    diagram.connect("source.x", "gain.x")
    diagram.connect("gain.y", "sum.a")
    diagram.connect("source.bias", "sum.b")

    signals = diagram.evaluate(0.0, {"source.x": 3.0, "source.bias": 1.0})

    assert signals["gain.y"] == 6.0
    assert signals["sum.y"] == 7.0


def test_pej_block_simulation_matches_motor_residual_signal():
    block_sim = build_pej_block_simulation(
        spring_model=FixedSpringModel(stiffness=2.0),
        controller=FeedforwardTorqueController(lambda t: 1.0),
        q_input=lambda t: 0.0,
        config=JointSimulationConfig(inertia=1.0, t_span=(0.0, 0.1), dt=0.01, initial_theta=0.2),
    )
    result = block_sim.simulate()
    signals = block_sim.sample_signals(result, index=0)

    assert signals["spring_model.tau_spring"] == 0.4
    assert signals["motor_model.tau_motor"] == 0.6
