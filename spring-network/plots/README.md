# Current figures

Only figures that directly support the active results are retained.

## `current/stiffness`

- `linear_heldout_examples.png` — held-out torque examples for the converged
  linear adaptive-stiffness result.
- `linear_training_convergence.png` — corresponding training history.
- `cubic_heldout_examples.png` — matched cubic-spring held-out examples.
- `cubic_training_convergence.png` — corresponding training history.

## `current/preload`

- `net_energy_command_schedule.png` — converged adaptive-preload command
  schedule.
- `net_energy_training_convergence.png` — net-energy training history.
- `time_and_torque_angle_examples.png` — representative preload behavior in
  time and torque-angle coordinates.
- `preload_topology.png` — active preload topology diagram.

## `current/profiles`

- `synthetic_torque_profile_gallery.png` — representative generated target
  torque-angle profiles.

## `current/validation`

- `linear_relaxation_convergence_audit.png` — linear mechanics convergence
  audit.
- `cubic_relaxation_convergence_audit.png` — cubic mechanics convergence
  audit.

## `current/spatial`

- `candidate131_56spring_preliminary_heldout_examples.png` — preliminary
  held-out examples for the current best-found 56-spring search candidate.
- `candidate131_56spring_preliminary_convergence.png` — its short screening
  run, not a paper-quality 5000-iteration result.
- `candidate022_48spring_dynamic_demo.html` — interactive 3D demonstration
  with dynamic learned stiffness and torque-time visualization. It uses the
  earlier 48-spring candidate and is retained as a visualization, not as the
  current topology winner.

Training and rendering scripts may generate additional files during future
experiments. Promote only final, interpretable figures into `current/`; delete
screening exports once their numerical tables have been preserved.
