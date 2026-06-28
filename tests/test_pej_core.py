import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pej


def test_table_4_offload_matches_reported_values_after_rounding():
    tables = pej.paper_tables()

    co_design = pej.offload_percentage(tables.power.co_design.before, tables.power.co_design.after)
    reference = pej.offload_percentage(tables.power.reference.before, tables.power.reference.after)

    np.testing.assert_allclose(np.round(co_design, 1), tables.power.co_design.offload_reported, atol=0.25)
    np.testing.assert_allclose(np.round(reference, 1), tables.power.reference.offload_reported, atol=0.25)


def test_sliding_mean_is_causal():
    result = pej.sliding_mean(np.array([1.0, 2.0, 3.0, 4.0]), window_length=3)

    np.testing.assert_allclose(result, np.array([1.0, 1.5, 2.0, 3.0]))


def test_piecewise_profile_ramps_to_zero_outside_active_range():
    profile = pej.make_piecewise_profile(0.0, 1.0, np.array([2.0, 4.0]), ramp_margin_rad=0.1)

    assert profile.theta[0] == -0.1
    assert profile.theta[-1] == 1.1
    np.testing.assert_allclose(profile.tau, np.array([0.0, 2.0, 4.0, 0.0]))
