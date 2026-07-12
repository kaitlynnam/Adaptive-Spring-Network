import numpy as np


ANGLE_LIMIT_RAD = np.deg2rad(44.0)
DEFAULT_PROFILE_SEED = 17
DEFAULT_TORQUE_LIMIT_NM = 115.0
TERRAIN_FAMILIES = ("flat_terrain", "rough_terrain", "mixed_terrain")


def generate_profile_parameters(rng, count):
    """Generate random piecewise-linear torque profiles and motion parameters."""
    profiles = []
    for index in range(count):
        profiles.append(base_profile_parameters(rng, f"piecewise_{index:04d}", "piecewise_linear"))
        profiles[-1]["knots_tau"] = random_torque_knots(rng, profiles[-1]["knots_theta"])
    return profiles


def generate_terrain_profile_parameters(rng, profiles_per_family, families=TERRAIN_FAMILIES):
    """Generate separated terrain-family piecewise-linear torque profiles."""
    profiles = []
    for family in families:
        for index in range(profiles_per_family):
            name = f"{family}_{index:04d}"
            profile = base_profile_parameters(rng, name, family)
            profile["knots_tau"] = terrain_torque_knots(rng, profile["knots_theta"], family)
            profiles.append(profile)
    rng.shuffle(profiles)
    return profiles


def base_profile_parameters(rng, name, family):
    motion = terrain_motion_parameters(rng, family)
    return {
        "name": name,
        "family": family,
        "knots_theta": random_theta_knots(rng),
        "knots_tau": None,
        **motion,
    }


def terrain_motion_parameters(rng, family):
    if family == "flat_terrain":
        return {
            "amplitude_deg": rng.uniform(16.0, 30.0),
            "frequency_hz": rng.uniform(0.55, 1.05),
            "phase": rng.uniform(0.0, 2.0 * np.pi),
            "harmonic_fraction": rng.uniform(0.03, 0.12),
            "bump_count": int(rng.integers(0, 3)),
            "noise_scale": rng.uniform(0.001, 0.007),
        }
    if family == "rough_terrain":
        return {
            "amplitude_deg": rng.uniform(22.0, 40.0),
            "frequency_hz": rng.uniform(0.70, 1.50),
            "phase": rng.uniform(0.0, 2.0 * np.pi),
            "harmonic_fraction": rng.uniform(0.12, 0.30),
            "bump_count": int(rng.integers(5, 11)),
            "noise_scale": rng.uniform(0.014, 0.034),
        }
    if family == "mixed_terrain":
        return {
            "amplitude_deg": rng.uniform(18.0, 36.0),
            "frequency_hz": rng.uniform(0.55, 1.30),
            "phase": rng.uniform(0.0, 2.0 * np.pi),
            "harmonic_fraction": rng.uniform(0.08, 0.22),
            "bump_count": int(rng.integers(2, 7)),
            "noise_scale": rng.uniform(0.006, 0.022),
        }
    return {
        "amplitude_deg": rng.uniform(18.0, 38.0),
        "frequency_hz": rng.uniform(0.55, 1.35),
        "phase": rng.uniform(0.0, 2.0 * np.pi),
        "harmonic_fraction": rng.uniform(0.05, 0.24),
        "bump_count": int(rng.integers(2, 8)),
        "noise_scale": rng.uniform(0.004, 0.026),
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


def terrain_torque_knots(rng, theta_knots, family):
    if family == "flat_terrain":
        stiffness = rng.uniform(55.0, 95.0)
        noise = rng.normal(0.0, 5.0, size=len(theta_knots))
        torque = -stiffness * theta_knots + noise
    elif family == "rough_terrain":
        stiffness = rng.uniform(115.0, 175.0)
        cubic = rng.uniform(35.0, 95.0)
        noise = rng.normal(0.0, 16.0, size=len(theta_knots))
        torque = -stiffness * theta_knots - cubic * theta_knots**3 + noise
    elif family == "mixed_terrain":
        k_neg = rng.uniform(75.0, 125.0)
        k_pos = rng.uniform(95.0, 155.0)
        cubic = rng.uniform(8.0, 55.0)
        stiffness = np.where(theta_knots < 0.0, k_neg, k_pos)
        noise = rng.normal(0.0, 10.0, size=len(theta_knots))
        torque = -stiffness * theta_knots - cubic * theta_knots**3 + noise
    else:
        return random_torque_knots(rng, theta_knots)

    return np.clip(torque, -DEFAULT_TORQUE_LIMIT_NM, DEFAULT_TORQUE_LIMIT_NM)


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
