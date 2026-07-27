from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from profile_generator import (  # noqa: E402
    TERRAIN_FAMILIES,
    default_profile_named,
    generate_classified_profile_parameters,
    generate_profile_parameters,
    profile_descriptor,
    profile_roughness_score,
)
from periodicity_classifier import periodicity_score  # noqa: E402


class ProfileGeneratorTests(unittest.TestCase):
    def test_seeded_arbitrary_generation_is_reproducible(self):
        first = generate_profile_parameters(np.random.default_rng(17), 10)
        second = generate_profile_parameters(np.random.default_rng(17), 10)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left["knots_theta"], right["knots_theta"])
            np.testing.assert_array_equal(left["knots_tau"], right["knots_tau"])

    def test_linear_sinusoidal_and_zigzag_have_increasing_roughness(self):
        theta = np.linspace(-1.0, 1.0, 5)
        linear = {"knots_theta": theta, "knots_tau": -theta}
        sinusoidal = {
            "knots_theta": theta,
            "knots_tau": np.sin(np.linspace(-np.pi, np.pi, 5)),
        }
        zigzag = {
            "knots_theta": theta,
            "knots_tau": np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        }
        self.assertLess(profile_roughness_score(linear), profile_roughness_score(sinusoidal))
        self.assertLess(profile_roughness_score(sinusoidal), profile_roughness_score(zigzag))

    def test_classification_is_balanced_and_ordered_by_shape_score(self):
        profiles = generate_classified_profile_parameters(np.random.default_rng(11), 20)
        grouped = {
            family: [profile for profile in profiles if profile["family"] == family]
            for family in TERRAIN_FAMILIES
        }
        self.assertEqual([len(grouped[family]) for family in TERRAIN_FAMILIES], [20, 20, 20])
        self.assertLessEqual(
            max(profile["roughness_score"] for profile in grouped["flat_terrain"]),
            min(profile["roughness_score"] for profile in grouped["mixed_terrain"]),
        )
        self.assertLessEqual(
            max(profile["roughness_score"] for profile in grouped["mixed_terrain"]),
            min(profile["roughness_score"] for profile in grouped["rough_terrain"]),
        )

    def test_named_gallery_profiles_are_not_limited_to_first_three(self):
        profile = default_profile_named("piecewise_0008")
        self.assertEqual(profile["name"], "piecewise_0008")

    def test_profile_descriptor_contains_five_angles_and_five_torques(self):
        profile = default_profile_named("piecewise_0000")
        descriptor = profile_descriptor(profile)
        self.assertEqual(descriptor.shape, (10,))
        self.assertTrue(np.all(np.isfinite(descriptor)))

    def test_periodicity_score_prefers_repeated_cycles(self):
        t = np.linspace(0.0, 6.0, 1201)
        repeated = np.sin(2.0 * np.pi * t)
        irregular = repeated + 0.7 * np.sin(2.0 * np.pi * 0.17 * t)
        repeated_score = periodicity_score(t, repeated, 2.0 * repeated, 1.0)
        irregular_score = periodicity_score(t, irregular, 2.0 * irregular, 1.0)
        self.assertGreater(
            repeated_score["periodicity_score"], irregular_score["periodicity_score"]
        )


if __name__ == "__main__":
    unittest.main()
