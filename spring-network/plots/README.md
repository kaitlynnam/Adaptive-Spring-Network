# Active figures

`period_adaptive_3d/` contains figures and interactive HTML for the causal
one-period-buffer controller. Non-period-controller figures are archived under
`../archive/non_period_buffer_20260811/artifacts/plots/` or the older archive
branches.

The bounded-extended checkpoint is the current broad-evaluation reference.

Main outputs use concise stable names:

- `fig01_topology.png`
- `fig02_causal_period_pipeline.png`
- `fig04a_deployment_torque_time.png`
- `fig04b_deployment_torque_angle.png`
- `fig04c_deployment_stiffness.png`
- `fig05a_many_profile_benchmark.png`
- `fig05b_many_profile_examples.png`
- `fig07_training_convergence.png`
- `interactive_simulation.html`

Figure 3 was retired because the held-out examples duplicated the clearer
multi-period deployment behavior in Figure 4. Figure 6 remains reserved for
the final mechanical-convergence output.

## Figure-generation standard

Whenever figures are generated for the most recent training run, include an
equilibrium-force residual figure as a standard output. Report the residual in
newtons and show its distribution or progression over the relevant held-out
samples or periods. At minimum, label the mean and maximum residual; use a log
scale when the range spans multiple orders of magnitude. Generate the figure
from the same exact relaxed-mechanics evaluation used for the other reported
test metrics, rather than from the differentiable training surrogate.
