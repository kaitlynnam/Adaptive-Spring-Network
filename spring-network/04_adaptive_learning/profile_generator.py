import numpy as np


ANGLE_LIMIT_RAD = np.deg2rad(44.0)
DEFAULT_PROFILE_SEED = 17
DEFAULT_TORQUE_LIMIT_NM = 115.0
TERRAIN_FAMILIES = ("flat_terrain", "mixed_terrain", "rough_terrain")
ROUGHNESS_ORDER = TERRAIN_FAMILIES


def generate_profile_parameters(rng, count):
    """Generate random piecewise-linear torque profiles and motion parameters."""
    profiles = []
    for index in range(count):
        profiles.append(base_profile_parameters(rng, f"piecewise_{index:04d}", "piecewise_linear"))
        profiles[-1]["knots_tau"] = random_torque_knots(rng, profiles[-1]["knots_theta"])
    return profiles


def generate_terrain_profile_parameters(rng, profiles_per_family, families=TERRAIN_FAMILIES):
    """Generate one random population, then classify it by relative roughness."""
    families = tuple(families)
    unknown = set(families) - set(ROUGHNESS_ORDER)
    if unknown:
        raise ValueError(f"Unsupported terrain families: {', '.join(sorted(unknown))}")

    family_order = [family for family in ROUGHNESS_ORDER if family in families]
    total = profiles_per_family * len(family_order)
    profiles = []
    for index in range(total):
        profile = base_profile_parameters(rng, f"candidate_{index:04d}", "unclassified")
        profile["knots_tau"] = random_restoring_torque_knots(rng, profile["knots_theta"])
        profile["roughness_score"] = profile_roughness_score(profile)
        profiles.append(profile)

    profiles.sort(key=lambda profile: profile["roughness_score"])
    classified = []
    for family_index, family in enumerate(family_order):
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


def random_restoring_torque_knots(rng, theta_knots):
    """Sample a random restoring curve before any terrain classification."""
    k_neg = rng.uniform(50.0, 175.0)
    k_pos = rng.uniform(50.0, 175.0)
    cubic = rng.uniform(0.0, 95.0)
    noise_scale = rng.uniform(0.0, 18.0)
    stiffness = np.where(theta_knots < 0.0, k_neg, k_pos)
    torque = -stiffness * theta_knots - cubic * theta_knots**3
    torque += rng.normal(0.0, noise_scale, size=len(theta_knots))
    return np.clip(torque, -DEFAULT_TORQUE_LIMIT_NM, DEFAULT_TORQUE_LIMIT_NM)


def profile_roughness_score(profile):
    """Score relative motion irregularity and torque-curve variation in [rougher = larger]."""
    theta = np.asarray(profile["knots_theta"], dtype=float)
    torque = np.asarray(profile["knots_tau"], dtype=float)
    slopes = np.diff(torque) / np.maximum(np.diff(theta), 1e-9)

    motion_score = np.mean(
        [
            (profile["frequency_hz"] - 0.55) / (1.50 - 0.55),
            (profile["harmonic_fraction"] - 0.03) / (0.30 - 0.03),
            profile["bump_count"] / 10.0,
            (profile["noise_scale"] - 0.001) / (0.034 - 0.001),
        ]
    )
    slope_magnitude = np.clip(np.mean(np.abs(slopes)) / 250.0, 0.0, 1.0)
    slope_variation = np.clip(np.std(slopes) / 250.0, 0.0, 1.0)
    torque_range = np.ptp(torque) / (2.0 * DEFAULT_TORQUE_LIMIT_NM)
    torque_score = np.mean([slope_magnitude, slope_variation, torque_range])
    return float(0.6 * motion_score + 0.4 * torque_score)


def profile_torque(theta, params):
    """Evaluate a randomized piecewise-linear torque-angle curve."""
    return np.interp(
        theta,
        params["knots_theta"],
        params["knots_tau"],
        left=params["knots_tau"][0],
        right=params["knots_tau"][-1],
    )


def default_piecewise_profiles(theta, count=3, seed=DEFAULT_PROFILE_SEED):
    rng = np.random.default_rng(seed)
    params = generate_profile_parameters(rng, count)
    return {profile["name"]: profile_torque(theta, profile) for profile in params}


def default_profile_named(name, count=3, seed=DEFAULT_PROFILE_SEED):
    rng = np.random.default_rng(seed)
    params = generate_profile_parameters(rng, count)
    for profile in params:
        if profile["name"] == name:
            return profile
    options = ", ".join(profile["name"] for profile in params)
    raise ValueError(f"Unknown profile {name!r}. Options: {options}")
