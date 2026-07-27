# Mechanically Adaptive PEJ Research Workspace

This repo contains Python tools for studying adaptive Parallel Elastic Joint
concepts. It starts from the equations in:

`Physical Imitation Learning: Distilling Control Policies into Passive Elasticity`

The current focus is offline research: load or generate joint trajectories,
estimate passive spring assistance, compare motor energy savings, and prototype
mechanism-inspired adaptive PEJ designs.

## Layout

- `python/pej`: reusable PEJ math package.
  - torque decomposition
  - positive-only motor power
  - offload percentage
  - Cost of Transport helpers
  - piecewise PEJ profile distillation
  - adaptive profile blending
  - actuator-tuned stiffness helpers
  - cam radius mapping
- `simulation`: reusable simulation framework.
  - SciPy joint dynamics
  - spring models
  - actuator dynamics
  - cam-controlled 3-spring network
  - energy accounting
  - executable block-style simulation helpers
  - plotting and SVG block diagram generation utilities
- `examples`: runnable examples grouped by topic.
  - `examples/paper`
  - `examples/profiles`
  - `examples/adaptive`
  - `examples/simulation`
  - `examples/cam`
- `tests`: regression tests for the math, simulation, cam network, and examples.
- `docs`: source notes and integration docs.
- `notebooks`: optional Jupyter experiments.
- `artifacts`: generated plots, tables, and diagrams. This folder is ignored by Git.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Common Commands

Reproduce paper-equation checks:

```sh
.venv/bin/python examples/paper/reproduce_paper_math.py
```

Distill a PEJ profile from trajectory data:

```sh
.venv/bin/python examples/profiles/distill_from_trajectory.py \
  --output-profile artifacts/profiles/synthetic_front_thigh_profile.csv
```

Compare adaptive spring models:

```sh
.venv/bin/python examples/adaptive/adaptive_spring_network.py
.venv/bin/python examples/adaptive/actuator_tuned_pej.py
```

Run the executable block-style simulation example:

```sh
.venv/bin/python examples/simulation/block_simulation.py
```

Generate SVG block diagrams:

```sh
.venv/bin/python examples/simulation/generate_block_diagrams.py
```

Run the cam-controlled 3-spring PEJ study:

```sh
.venv/bin/python examples/cam/cam_spring_network.py
.venv/bin/python examples/cam/cam_spring_sensitivity.py
```

Generated outputs are written under `artifacts/`.

Comparison examples print the same core measurement categories where they
apply:

- `scenario`
- `case`
- `motor_energy_j`
- `mean_power_w`
- `cam_energy_j`
- `net_saved_j`
- `offload_pct`
- tuning/mechanism fields such as `mean_q`, `mean_phi`, and `mean_k`
- spring strength fields such as `spring_k`, `spring_k_1`, `spring_k_2`, and `spring_k_3`

The same tables are written as CSV files under `artifacts/tables/`.

## Trajectory Data

Trajectory inputs may be CSV or NPZ. Required columns/arrays are:

- `time`
- `joint_name`
- `theta`
- `theta_dot`
- `tau_total`

Optional metadata columns/arrays are:

- `terrain`
- `policy`
- `robot_id`

Important convention:

```text
tau_total = active joint torque demand before passive spring assistance
```

This keeps the repo easy to connect to any simulator later. Export the original
controller/actuator torque demand, then use this repo offline to evaluate PEJ
profiles and cam spring networks.

See `docs/simulator_trajectory_schema.md` for the integration contract.

## Tests

```sh
.venv/bin/python -m pytest
```

The implementation uses NumPy, SciPy, and Matplotlib.
