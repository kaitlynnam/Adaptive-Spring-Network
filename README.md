# Mechanically Adaptive PEJ Research Workspace

This workspace starts by reproducing the math from the Bristol PEJ paper:

`Physical Imitation Learning: Distilling Control Policies into Passive Elasticity`

Current contents:

- `docs/paper_equations.md`: equations, constants, and appendix values transcribed from the PDF.
- `python/pej`: Python functions for PEJ torque decomposition, positive-only motor power, CoT, offload, tracking error, piecewise PEJ profiles, distillation, and cam mapping.
- `python/scripts/demo_reproduce_paper_math.py`: a runnable Python script that checks the paper's Table 4 offload values and exercises the equation layer on synthetic data.
- `python/scripts/distill_from_trajectory.py`: a runnable trajectory-to-PEJ pipeline that loads rollout data, distills a torque-angle profile, estimates offload, and maps the profile to a cam radius curve.

Run with Python:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python python/scripts/demo_reproduce_paper_math.py
.venv/bin/python python/scripts/distill_from_trajectory.py --output-profile docs/generated/synthetic_front_thigh_profile.csv
```

Trajectory inputs may be CSV or NPZ. Required columns/arrays are:

- `time`
- `joint_name`
- `theta`
- `theta_dot`
- `tau_total`

Optional metadata columns/arrays are `terrain`, `policy`, and `robot_id`. If no input is provided,
`distill_from_trajectory.py` uses a deterministic synthetic `front_thigh` rollout.

Run tests:

```sh
.venv/bin/python -m pytest
```

The implementation uses NumPy for numerical computation.
