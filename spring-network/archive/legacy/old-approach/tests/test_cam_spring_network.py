import numpy as np

from simulation.cam_spring_network import (
    CamSpring,
    CamSpringNetworkLimits,
    CamSpringNetwork,
    fit_spring_rates_to_targets,
    optimize_spring_rates_for_energy,
    positive_cam_power,
    smooth_engagement,
)
from python.pej.core import motor_power
from simulation.energy import net_energy_savings


def test_engagement_is_zero_before_phi_start():
    assert smooth_engagement(-0.1, 0.0, 1.0) == 0.0


def test_engagement_is_one_after_phi_end():
    assert smooth_engagement(1.5, 0.0, 1.0) == 1.0


def test_engagement_is_between_zero_and_one_inside_range():
    assert smooth_engagement(0.25, 0.0, 1.0) == 0.25


def test_spring_compression_is_never_negative():
    network = CamSpringNetwork(
        [CamSpring(k=100.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.01)]
    )

    compression = network.spring_compressions(theta=-10.0, phi=0.0)

    np.testing.assert_allclose(compression, np.array([0.0]))


def test_spring_force_increases_with_cam_angle():
    network = CamSpringNetwork(
        [CamSpring(k=100.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.01)]
    )

    low_force = network.spring_forces(theta=0.0, phi=0.1)[0]
    high_force = network.spring_forces(theta=0.0, phi=0.9)[0]

    assert high_force > low_force


def test_total_spring_torque_magnitude_increases_as_more_springs_engage():
    network = CamSpringNetwork(
        [
            CamSpring(k=100.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=0.5, max_compression=0.01),
            CamSpring(k=200.0, preload=0.0, lever_arm=0.1, phi_start=0.5, phi_end=1.0, max_compression=0.01),
        ]
    )

    low_torque = network.torque(theta=0.0, phi=0.25)
    high_torque = network.torque(theta=0.0, phi=1.0)

    assert abs(high_torque) > abs(low_torque)


def test_cam_actuator_power_is_positive_only():
    power = positive_cam_power(np.array([1.0, 1.0]), np.array([2.0, -2.0]))

    np.testing.assert_allclose(power, np.array([2.0, 0.0]))


def test_net_energy_saved_subtracts_cam_actuator_energy():
    time = np.array([0.0, 1.0])
    summary = net_energy_savings(
        time,
        motor_power_without_spring=np.array([10.0, 10.0]),
        motor_power_with_spring=np.array([4.0, 4.0]),
        cam_actuator_power=np.array([1.5, 1.5]),
    )

    assert summary.motor_energy_saved == 6.0
    assert summary.cam_actuator_energy == 1.5
    assert summary.net_energy_saved == 4.5
    assert summary.net_offload_percentage == 45.0


def test_theta_rest_and_assist_direction_define_restoring_torque_sign():
    spring = CamSpring(k=100.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.0)
    positive_assist = CamSpringNetwork([spring], theta_rest=0.2, assist_direction=1.0)
    negative_assist = CamSpringNetwork([spring], theta_rest=0.2, assist_direction=-1.0)

    assert positive_assist.torque(theta=0.4, phi=0.0) < 0.0
    assert negative_assist.torque(theta=0.0, phi=0.0) > 0.0


def test_cam_torque_matches_spring_energy_derivative_inside_engagement():
    network = CamSpringNetwork(
        [CamSpring(k=100.0, preload=0.001, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.01)]
    )

    assert abs(network.cam_torque_energy_error(theta=0.2, phi=0.5)) < 1e-8


def test_constraint_report_identifies_violations():
    network = CamSpringNetwork(
        [CamSpring(k=100.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.02)]
    )

    report = network.check_constraints(
        theta=np.array([0.5]),
        phi=np.array([1.0]),
        phi_dot=np.array([2.0]),
        limits=CamSpringNetworkLimits(
            max_compression=0.01,
            max_force=1.0,
            max_abs_spring_torque=0.1,
            max_abs_cam_torque=0.1,
            max_abs_phi_speed=1.0,
        ),
    )

    assert not report.passed
    assert "max_compression" in report.violations
    assert "max_abs_phi_speed" in report.violations


def test_fit_spring_rates_to_target_torque_recovers_single_spring_rate():
    template = CamSpringNetwork(
        [CamSpring(k=1.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.0)]
    )
    theta = np.array([0.1, 0.2, 0.3])
    phi = np.zeros_like(theta)
    target = CamSpringNetwork(
        [CamSpring(k=250.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.0)]
    ).torque(theta, phi)

    fit = fit_spring_rates_to_targets(template, theta, phi, target, max_k=1000.0)

    np.testing.assert_allclose(fit.fitted_spring_rates, np.array([250.0]), rtol=1e-6)
    assert fit.rms_error < 1e-10


def test_optimize_spring_rates_for_energy_reduces_objective():
    template = CamSpringNetwork(
        [CamSpring(k=0.0, preload=0.0, lever_arm=0.1, phi_start=0.0, phi_end=1.0, max_compression=0.0)]
    )
    time = np.linspace(0.0, 1.0, 51)
    theta = 0.2 + 0.05 * np.sin(2.0 * np.pi * time)
    theta_dot = np.gradient(theta, time)
    phi = np.zeros_like(time)
    phi_dot = np.zeros_like(time)
    tau_required = -2.0 * theta
    baseline = motor_power(tau_required, theta_dot)

    result = optimize_spring_rates_for_energy(
        template,
        time,
        theta,
        theta_dot,
        phi,
        phi_dot,
        tau_required,
        baseline,
        initial_k=np.array([0.0]),
        max_k=1000.0,
    )

    assert result.success
    assert result.net_energy_saved >= 0.0
