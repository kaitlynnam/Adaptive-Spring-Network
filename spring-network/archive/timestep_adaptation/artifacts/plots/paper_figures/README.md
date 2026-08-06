# Paper figure candidates

1. `fig01_3d_adaptive_spring_network.png` — current best-found 56-spring
   topology at +25 degrees. This is a preliminary topology-search result.
2. `fig02_neural_to_spring_pipeline.png` — causal information flow from
   motion/history inputs through the neural controller and relaxed 3D spring
   mechanics.
3. `fig03a_torque_angle_profiles.png` and `fig03b_torque_time_profiles.png` —
   three unclassified profiles from the regenerated held-out dataset, using
   a five-cycle, 20-second trajectory at 0.25 Hz using the test-report style.
4. `fig04_example_adaptive_behavior.png` — one actual held-out 48-spring test
   rollout showing
   target, spring, and residual motor torque, joint motion, and all commanded
   stiffnesses. Its middle panel compares the piecewise-linear target and
   relaxed spring torque versus joint angle. Each heatmap row is one physical
   spring, ordered by neutral-position midpoint proximity so nearby rows tend
   to represent springs in nearby regions of the mechanism. All three panels
   use the same causal rollout. This is not the current 56-spring winner.
5. `fig05_primary_performance_comparison_preliminary.png` — fixed versus
   adaptive mechanics for candidate 131. It uses a short 300-iteration screen
   and 30 held-out profiles, so it must be labeled preliminary.
6. A multiple-seed robustness figure is **not available yet**. It requires the
   independent 5000-iteration runs documented in `spring-network/PAPER_RUNS.md`.
7. `fig06_offload_relaxation_convergence.png` — mean offload versus mechanics
   relaxation depth for the converged 2D linear and cubic experiments.

The requested list contained two figures numbered 5. Recommended final
numbering is: primary comparison as Figure 5, seed robustness as Figure 6,
and mechanical convergence as Figure 7. Rename the files only after the
multi-seed experiment is complete.
