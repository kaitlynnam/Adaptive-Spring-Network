from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "04_adaptive_learning"))

from profile_generator import (  # noqa: E402
    default_profile_named,
    generate_profile_parameters,
    profile_descriptor,
)


class ProfileGeneratorTests(unittest.TestCase):
    def test_seeded_arbitrary_generation_is_reproducible(self):
        first = generate_profile_parameters(np.random.default_rng(17), 10)
        second = generate_profile_parameters(np.random.default_rng(17), 10)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left["knots_theta"], right["knots_theta"])
            np.testing.assert_array_equal(left["knots_tau"], right["knots_tau"])

    def test_named_gallery_profiles_are_not_limited_to_first_three(self):
        profile = default_profile_named("piecewise_0008")
        self.assertEqual(profile["name"], "piecewise_0008")

    def test_profile_descriptor_contains_five_angles_and_five_torques(self):
        profile = default_profile_named("piecewise_0000")
        descriptor = profile_descriptor(profile)
        self.assertEqual(descriptor.shape, (10,))
        self.assertTrue(np.all(np.isfinite(descriptor)))

if __name__ == "__main__":
    unittest.main()
