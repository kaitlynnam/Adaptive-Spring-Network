import numpy as np


ANGLE_LIMIT_RAD = np.deg2rad(45.0)
DEFAULT_PROFILE_SEED = 17
DEFAULT_TORQUE_LIMIT_NM = 115.0
TERRAIN_FAMILIES = ("flat_terrain", "mixed_terrain", "rough_terrain")
PROFILE_CLASSIFICATION = "arbitrary_shape_roughness_v1"


def generate_profile_parameters(rng, count):
    """Generate random piecewise-linear torque profiles and motion parameters."""
    profiles = []
    for index in range(count):
        profiles.append(base_profile_parameters(rng, f"piecewise_{index:04d}", "piecewise_linear"))
        profiles[-1]["knots_tau"] = random_torque_knots(rng, profiles[-1]["knots_theta"])
    return profiles


def generate_classified_profile_parameters(rng, profiles_per_family):
    """Generate arbitrary profiles and split them into relative shape-roughness thirds.

    The terrain-like names are synthetic shape labels, not labels inferred from
    measured terrain. Every torque knot is sampled independently before any
    classification takes place.
    """
    if profiles_per_family <= 0:
        raise ValueError("profiles_per_family must be positive")

    total = profiles_per_family * len(TERRAIN_FAMILIES)
    profiles = generate_profile_parameters(rng, total)
    for profile in profiles:
        profile["roughness_score"] = profile_roughness_score(profile)
        profile["classification"] = PROFILE_CLASSIFICATION

    profiles.sort(key=lambda profile: profile["roughness_score"])
    classified = []
    for family_index, family in enumerate(TERRAIN_FAMILIES):
        start = family_index * profiles_per_family
        stop = start + profiles_per_family
        for index, profile in enumerate(profiles[start:stop]):
            profile["family"] = family
            profile["name"] = f"{family}_{index:04d}"
            classified.append(profile)

    rng.shuffle(classified)
    return classified


def base_profile_parameters(rng, name, family):
    motion = random_motion_parameters(rng)
    return {
        "name": name,
        "family": family,
        "knots_theta": random_theta_knots(rng),
        "knots_tau": None,
        **motion,
    }


def random_motion_parameters(rng):
    """Sample motion without assigning a terrain label."""
    return {
        "amplitude_deg": rng.uniform(16.0, 40.0),
        "frequency_hz": rng.uniform(0.55, 1.50),
        "phase": rng.uniform(0.0, 2.0 * np.pi),
        "harmonic_fraction": rng.uniform(0.03, 0.30),
        "bump_count": int(rng.integers(0, 11)),
        "noise_scale": rng.uniform(0.001, 0.034),
    }


def random_theta_knots(rng):
    """Five target points, with first and last angles fixed at the bounds."""
    interior = rng.uniform(-ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD, size=3)
    knots = np.concatenate(([-ANGLE_LIMIT_RAD], np.sort(interior), [ANGLE_LIMIT_RAD]))
    return np.asarray(knots, dtype=float)


def random_torque_knots(rng, theta_knots):
    del theta_knots
    normalized = rng.uniform(-1.0, 1.0, size=5)
    return DEFAULT_TORQUE_LIMIT_NM * normalized


def profile_roughness_score(profile):
    """Score arbitrary torque-curve complexity relative to linear behavior.

    A straight line scores zero. Smooth single-bend or sinusoidal-like curves
    score between straight lines and repeatedly reversing zigzags. The score
    intentionally excludes motion parameters so classification describes the
    torque-angle profile itself.
    """
    theta = np.asarray(profile["knots_theta"], dtype=float)
    torque = np.asarray(profile["knots_tau"], dtype=float)
    slopes = np.diff(torque) / np.maximum(np.diff(theta), 1e-9)
    torque_range = max(float(np.ptp(torque)), 1e-9)

    # Closed-form least-squares line. This is equivalent to ``polyfit(..., 1)``
    # here, while avoiding an unnecessary LAPACK/OpenMP dependency for five
    # scalar knots.
    centered_theta = theta - np.mean(theta)
    slope = np.sum(centered_theta * (torque - np.mean(torque))) / max(
        float(np.sum(centered_theta**2)), 1e-12
    )
    line = np.mean(torque) + slope * centered_theta
    linear_error = np.clip(np.sqrt(np.mean((torque - line) ** 2)) / torque_range, 0.0, 1.0)

    total_variation_ratio = np.sum(np.abs(np.diff(torque))) / torque_range
    variation_excess = np.clip((total_variation_ratio - 1.0) / 3.0, 0.0, 1.0)

    slope_rms = max(float(np.sqrt(np.mean(slopes**2))), 1e-9)
    slope_variation = np.clip(np.std(slopes) / slope_rms, 0.0, 1.0)

    nonzero_signs = np.sign(slopes[np.abs(slopes) > 1e-9])
    reversals = np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1])
    reversal_fraction = reversals / max(len(slopes) - 1, 1)

    return float(
        0.35 * linear_error
        + 0.30 * variation_excess
        + 0.20 * slope_variation
        + 0.15 * reversal_fraction
    )


def profile_torque(theta, params):
    """Evaluate a randomized piecewise-linear torque-angle curve."""
    return np.interp(
        theta,
        params["knots_theta"],
        params["knots_tau"],
        left=params["knots_tau"][0],
        right=params["knots_tau"][-1],
    )


def profile_descriptor(params, torque_scale=DEFAULT_TORQUE_LIMIT_NM):
    """Return the commanded five-knot curve as ten normalized MLP inputs."""
    angle_features = np.asarray(params["knots_theta"], dtype=float) / ANGLE_LIMIT_RAD
    torque_features = np.asarray(params["knots_tau"], dtype=float) / max(float(torque_scale), 1e-9)
    return np.concatenate((angle_features, torque_features))


def default_piecewise_profiles(theta, count=3, seed=DEFAULT_PROFILE_SEED):
    rng = np.random.default_rng(seed)
    params = generate_profile_parameters(rng, count)
    return {profile["name"]: profile_torque(theta, profile) for profile in params}


def default_profile_named(name, count=3, seed=DEFAULT_PROFILE_SEED):
    if name.startswith("piecewise_"):
        try:
            count = max(count, int(name.removeprefix("piecewise_")) + 1)
        except ValueError:
            pass
    rng = np.random.default_rng(seed)
    params = generate_profile_parameters(rng, count)
    for profile in params:
        if profile["name"] == name:
            return profile
    options = ", ".join(profile["name"] for profile in params)
    raise ValueError(f"Unknown profile {name!r}. Options: {options}")
