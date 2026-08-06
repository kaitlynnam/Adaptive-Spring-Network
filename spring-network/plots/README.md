# Active passive-control figures

All active plots are for the profile-conditioned passive controller using
`topologies/spatial/internal_fan_3d_60_spring.json`.

Currently available from the exact-mechanics seed-101 checkpoint:

- `profile_conditioned_passive_3d/profile_passive_3d_60spring_seed101_torque_angle.png`
- `profile_conditioned_passive_3d/profile_passive_3d_60spring_seed101_time_traces.png`
- `profile_conditioned_passive_3d/profile_passive_3d_60spring_seed101_stiffness_heatmap.png`

## Required paper set

1. 60-spring 3D topology and labeled joint geometry.
2. Profile-to-fixed-stiffness passive-control pipeline.
3. Representative held-out torque-angle behavior.
4. Representative held-out time traces.
5. Per-spring stiffness allocation for representative profiles.
6. Aggregate held-out performance by roughness family and seed.
7. Training and mechanics-correction convergence.
8. Multi-seed robustness summary.

Items 3–5 are available. Items 6 and 8 must wait for the running seed-202 and
seed-303 jobs. Figures are intentionally not regenerated during those runs.
Figures belonging to the former 24-, 30-, 48-, and 56-spring pipelines are
stored under `archive/timestep_adaptation/artifacts/plots/`.
