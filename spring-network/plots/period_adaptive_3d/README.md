# Figure index

Updated 2026-10-01. The main figures are in this directory; each refreshed figure
has a transparent PNG and an SVG with the same filename stem.

| Figure | PNG | Contents |
| --- | --- | --- |
| 1 | [Topology](fig01_topology.png) | Active hybrid 60-spring topology |
| 2 | [Causal pipeline](fig02_causal_period_pipeline.png) | Completed-period observation and next-period stiffness |
| 3 | [Test-profile flow](fig03_test_profile_flow.png) | Collision-exact5refresh run, profile 629, original saved data |
| 4a | [Torque versus time](fig04a_deployment_torque_time.png) | Three target profiles, two periods each |
| 4b | [Torque versus angle](fig04b_deployment_torque_angle.png) | Target, spring, and residual motor torque for each period |
| 4c | [Stiffness schedule](fig04c_deployment_stiffness.png) | One fixed stiffness vector per period |
| 5a | [Benchmark](fig05a_many_profile_benchmark.png) | Original bounded-extended benchmark arrays |
| 5b | [Benchmark examples](fig05b_many_profile_examples.png) | Lowest, median, and highest offload profiles |
| 6 | [Force residual](fig06_equilibrium_force_residual.png) | Exact relaxed deployment mechanics, mean and maximum in N |
| 7 | [Training convergence](fig07_training_convergence.png) | Original image retained; original loss history unavailable |

## Other assets

- [Experiment figures](experiments/): regenerated benchmarks and retained training curves.
- [Previous experiment styling](experiments/previous_style/): original exports preserved during cleanup.
- [PowerPoint components](fig01_powerpoint_assets/): separate PNG/SVG components.
- [Extended abstract](Extended%20Abstract/): original assembled reference images, preserved.
- [Topology artwork](Spring%20Topology.png) and [Photoshop source](Spring%20Topology.psd): preserved user artwork.

Torque plots use dashed black targets, blue spring torque, and orange residual
motor torque. Default-period comparison curves remain gray. Redundant axes
titles are removed while labels, units, panel identities, and metrics remain.
The model/data for each existing figure is retained; these are not all results
from a single checkpoint. Archived research branches and existing animations
are historical outputs and have not been regenerated.

## Reproduce from the repository root

```powershell
python spring-network/04_adaptive_learning/refresh_publication_figures.py --all-experiments
python spring-network/04_adaptive_learning/deploy_period_adaptive_3d.py --trajectory-mode grouped --periods 6 --periods-per-profile 2 --device cpu
python spring-network/04_adaptive_learning/generate_period_pipeline_figure.py
python spring-network/04_adaptive_learning/generate_test_profile_flow_figure.py
python -c "import sys; sys.path.insert(0,'spring-network/01_core_model'); from render_spatial_topology import render; render()"
```

Benchmark refresh uses saved arrays without reevaluation or retraining. Deployment
uses the bounded-extended checkpoint, seed 501, and 300 relaxation steps; exact
arrays and force residuals are saved beside its CSV in
`../../tables/period_adaptive_3d/period_adaptive_3d_60spring_bounded_extended_deployment.npz`.
Future training exports save a `training_history.csv` alongside the figures so
convergence curves can be reproduced. No historical curve was inferred from pixels.
