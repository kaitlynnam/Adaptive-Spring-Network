import numpy as np

from simulation.controller import FeedforwardTorqueController
from simulation.dynamics import JointSimulationConfig, simulate_joint
from simulation.energy import summarize_energy
from simulation.spring_models import ActuatorTunedModel, FixedSpringModel


def test_simulate_joint_integrates_state_and_outputs_power():
    controller = FeedforwardTorqueController(lambda t: 0.1)
    config = JointSimulationConfig(inertia=1.0, t_span=(0.0, 0.2), dt=0.02)

    result = simulate_joint(
        spring_model=FixedSpringModel(stiffness=0.0),
        controller=controller,
        q_input=lambda t: 0.0,
        config=config,
    )

    assert result.time.size == 11
    assert result.theta[-1] > result.theta[0]
    assert result.theta_dot[-1] > result.theta_dot[0]
    assert np.all(result.motor_power >= 0.0)


def test_fixed_spring_reduces_motor_power_against_no_spring():
    controller = FeedforwardTorqueController(lambda t: 1.0)
    config = JointSimulationConfig(
        inertia=1.0,
        t_span=(0.0, 0.5),
        dt=0.01,
        initial_theta=0.2,
    )

    no_spring = simulate_joint(
        spring_model=FixedSpringModel(stiffness=0.0),
        controller=controller,
        q_input=lambda t: 0.0,
        config=config,
    )
    fixed = simulate_joint(
        spring_model=FixedSpringModel(stiffness=2.0),
        controller=controller,
        q_input=lambda t: 0.0,
        config=config,
    )

    assert np.mean(fixed.motor_power) < np.mean(no_spring.motor_power)
    summary = summarize_energy(
        fixed.time,
        fixed.tau_motor,
        fixed.theta_dot,
        baseline_power=no_spring.motor_power,
    )
    assert summary.offload_percentage > 0.0


def test_actuator_tuned_simulation_tracks_phi_state_with_time_constant():
    model = ActuatorTunedModel(
        k_soft=1.0,
        k_stiff=3.0,
        phi_min=0.0,
        phi_max=1.0,
        actuator_time_constant=0.1,
    )
    result = simulate_joint(
        spring_model=model,
        controller=FeedforwardTorqueController(lambda t: 0.0),
        q_input=lambda t: 1.0,
        config=JointSimulationConfig(inertia=1.0, t_span=(0.0, 0.3), dt=0.01, initial_phi=0.0),
    )

    assert result.phi[0] == 0.0
    assert result.phi[-1] > result.phi[0]
    assert result.phi[-1] < 1.0
    assert result.k_eff[-1] > result.k_eff[0]
