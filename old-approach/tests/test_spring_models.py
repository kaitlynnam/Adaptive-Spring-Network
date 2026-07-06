import numpy as np

from simulation.spring_models import AdaptiveBlendModel, ActuatorTunedModel, FixedSpringModel


def test_fixed_spring_model_computes_linear_torque():
    model = FixedSpringModel(stiffness=5.0, rest_angle=0.1)

    torque = model.compute_torque(np.array([0.1, 0.2, 0.3]))

    np.testing.assert_allclose(torque, np.array([0.0, 0.5, 1.0]))


def test_adaptive_blend_model_interpolates_profiles():
    model = AdaptiveBlendModel(
        flat_profile=lambda theta: 2.0 * theta,
        rough_profile=lambda theta: 6.0 * theta,
    )

    torque = model.compute_torque(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.5, 1.0]))

    np.testing.assert_allclose(torque, np.array([1.0, 2.0, 3.0]))


def test_actuator_tuned_model_maps_q_to_stiffness_and_torque():
    model = ActuatorTunedModel(k_soft=4.0, k_stiff=10.0, phi_min=-0.2, phi_max=0.4)

    q = np.array([0.0, 0.5, 1.0])
    torque = model.compute_torque(theta=np.array([0.2, 0.2, 0.2]), q=q)

    np.testing.assert_allclose(model.command(q).phi, np.array([-0.2, 0.1, 0.4]))
    np.testing.assert_allclose(model.effective_stiffness(q), np.array([4.0, 7.0, 10.0]))
    np.testing.assert_allclose(torque, np.array([0.8, 1.4, 2.0]))
