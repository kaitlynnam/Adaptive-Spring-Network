import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pej


def test_blend_profiles_uses_flat_profile_when_q_is_zero():
    theta = np.array([0.0, 0.5, 1.0])

    result = pej.blend_profiles(theta, lambda x: 2.0 * x, lambda x: 10.0 + x, q=0.0)

    np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0]))


def test_blend_profiles_uses_rough_profile_when_q_is_one():
    theta = np.array([0.0, 0.5, 1.0])

    result = pej.blend_profiles(theta, lambda x: 2.0 * x, lambda x: 10.0 + x, q=1.0)

    np.testing.assert_allclose(result, np.array([10.0, 10.5, 11.0]))


def test_blend_profiles_averages_profiles_when_q_is_half():
    theta = np.array([0.0, 0.5, 1.0])

    result = pej.blend_profiles(theta, lambda x: 2.0 * x, lambda x: 10.0 + x, q=0.5)

    np.testing.assert_allclose(result, np.array([5.0, 5.75, 6.5]))


def test_roughness_score_and_q_mapping():
    flat_score = pej.roughness_score(np.array([1.0, 1.0, 1.0]))
    rough_score = pej.roughness_score(np.array([-1.0, 0.0, 1.0]))

    assert flat_score == 0.0
    assert rough_score > flat_score
    np.testing.assert_allclose(
        pej.roughness_to_q(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0),
        np.array([0.0, 0.0, 0.5, 1.0, 1.0]),
    )


def test_actuator_tuned_stiffness_uses_soft_stiffness_when_q_is_zero():
    response = pej.actuator_tuned_stiffness(
        theta=0.2,
        q=0.0,
        k_soft=4.0,
        k_stiff=10.0,
        phi_min=-0.5,
        phi_max=0.5,
    )

    np.testing.assert_allclose(response.k_eff, 4.0)
    np.testing.assert_allclose(response.phi, -0.5)
    np.testing.assert_allclose(response.tau_spring, 0.8)


def test_actuator_tuned_stiffness_uses_stiff_stiffness_when_q_is_one():
    response = pej.actuator_tuned_stiffness(
        theta=0.2,
        q=1.0,
        k_soft=4.0,
        k_stiff=10.0,
        phi_min=-0.5,
        phi_max=0.5,
    )

    np.testing.assert_allclose(response.k_eff, 10.0)
    np.testing.assert_allclose(response.phi, 0.5)
    np.testing.assert_allclose(response.tau_spring, 2.0)


def test_actuator_tuned_stiffness_uses_midpoint_stiffness_when_q_is_half():
    response = pej.actuator_tuned_stiffness(
        theta=0.2,
        q=0.5,
        k_soft=4.0,
        k_stiff=10.0,
        phi_min=-0.5,
        phi_max=0.5,
    )

    np.testing.assert_allclose(response.k_eff, 7.0)
    np.testing.assert_allclose(response.phi, 0.0)
    np.testing.assert_allclose(response.tau_spring, 1.4)


def test_actuator_tuned_torque_increases_with_q_for_same_positive_theta():
    response = pej.actuator_tuned_stiffness(
        theta=np.array([0.2, 0.2, 0.2]),
        q=np.array([-1.0, 0.5, 2.0]),
        k_soft=4.0,
        k_stiff=10.0,
        phi_min=-0.5,
        phi_max=0.5,
    )

    np.testing.assert_allclose(response.q, np.array([0.0, 0.5, 1.0]))
    assert response.tau_spring[0] < response.tau_spring[1] < response.tau_spring[2]
