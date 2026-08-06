# Active 60-spring tables

The only active result directory is `profile_conditioned_passive_3d/`.
It contains results for `topologies/spatial/internal_fan_3d_60_spring.json`:

- exact-mechanics profile-conditioned passive checkpoints and seed results;
- mechanics-correction and convergence results for that topology;
- the matching 60-spring per-profile oracle upper bound.

Canonical files use `passive_60spring_seedNNN`. The older `rerelaxed`,
`refresh4`, and `full_refresh` names were experimental history and are archived.

The controller uses positive softplus stiffness with no imposed upper cap.
This is the current unconstrained upper-bound assumption; “unbounded” does not
mean negative stiffness or infinite values, and it is no longer placed in
active filenames.

Tables for other topologies, planar models, screens, IsaacLab, and legacy
experiments are under `archive/non_active_artifacts/`.
