# Spring Network Simulator Summary

This folder contains the active spring-network approach for modeling passive assistance around a single 2D revolute robot joint.

## Current Purpose

The project is a simple, debuggable simulator for a spring network placed around one revolute joint. It answers two main questions:

- What torque does a given spring topology produce as the joint moves?
- How much motor energy does that topology save over a time-varying joint trajectory?

The current focus is mechanical evaluation, not final hardware design. The model is still quasi-static and intentionally small.

## Current Model

The simulator models one revolute joint at the origin with two rigid limbs:

- Limb 1 points left and is treated as fixed.
- Limb 2 rotates by joint angle `theta`.
- Nodes are placed around the joint.
- Springs connect pairs of nodes.
- Spring forces are computed with Hooke's law.
- Net joint torque is computed from forces applied to limb-2 nodes.
- Internal nodes can now relax toward quasi-static mechanical equilibrium before torque is evaluated.

The model does not include multi-joint dynamics, heat transfer, closed-loop control, inertia, damping, or a full nonlinear mechanics solver.

## File Layout

- `01_core_model/`  
  Defines the mechanics layer: geometry, spring physics, network evaluation, topology loading, and plotting helpers.

- `02_baseline_profiles/`  
  Runs the baseline demo and baseline-only angle-domain profile inspection.

- `03_stiffness_optimization (optional)/`  
  Optional tooling for finite-difference stiffness optimization. Generated optimized-topology artifacts are not part of the current kept result set.

- `04_adaptive_learning/`  
  Trains adaptive stiffness models. The current kept model is the adaptive trained model.

- `05_trajectory_evaluation/`  
  Runs the preferred time-domain evaluator and trajectory-based model comparison. It simulates generated trajectories, computes spring torque at each timestep, computes motor power with and without the spring network, integrates energy, prints summaries, and saves CSV outputs.

## Current Topologies And Models

Topology files live in `topologies/`. Adaptive model files live in `models/`.

### `topologies/baseline_model.json`

This is the hand-picked baseline structure. It has:

- 10 nodes
- 16 springs
- fixed anchors
- limb-1 nodes
- limb-2 nodes
- internal nodes

The structure is intentionally asymmetric and manually chosen so the model is easy to inspect and debug. It is not a perfect grid or optimized web.

### `models/adaptive_trained_model.npz`

This is the active dataset-trained adaptive stiffness model. It no longer receives target torque as an input. Its inputs are only motion-history features:

```text
theta history
theta_dot history
theta_ddot history
```

The current saved model uses a 10-sample causal window, so its input size is 30 values:

```text
10 samples * (theta, theta_dot, theta_ddot) = 30 inputs
```

Flat-terrain training trajectories are smooth, rough-terrain trajectories include irregular bumps and smoothed noise, and mixed-terrain trajectories combine smooth and rough segments. The active model was trained on generated trajectories with 160 samples each. Time-domain evaluation uses a separate generated batch with 300 samples per trajectory.

Because it cannot see the requested torque profile directly, it infers useful stiffness changes from recent motion shape.

Additional saved trial variants from the window-size sweep live in `models/`:

```text
adaptive_trained_model_w5_trial.npz
adaptive_trained_model_w10_trial.npz
adaptive_trained_model_w15_trial.npz
adaptive_trained_model_w20_trial.npz
```

The active `adaptive_trained_model.npz` is the window-10 version.

## Terrain Torque Profiles

The current project uses terrain-style restoring torque profiles:

- `flat_terrain`
- `rough_terrain`
- `mixed_terrain`

The angle-domain profile functions are in `evaluate_profiles.py`:

```text
flat_terrain  = -75 * theta
rough_terrain = -135 * theta - 55 * theta^3
mixed_terrain = asymmetric left/right stiffness with an added cubic term
```

The time-domain evaluator also supports `mixed_terrain` with a velocity-dependent term:

```text
mixed_terrain = -90 * theta - 20 * theta^3 + 10 * sign(theta_dot) * theta^2
```

## Internal Node Relaxation

Internal nodes are no longer treated as permanently fixed during trajectory evaluation. When relaxation is enabled, the network:

1. Updates fixed, limb-1, and limb-2 nodes from the current joint angle.
2. Warm-starts internal nodes from their previous positions.
3. Minimizes total spring potential energy over the internal-node coordinates with SciPy `L-BFGS-B`.
4. Stops when the energy gradient, equivalent to negative net internal force, is small enough or the iteration limit is reached.
5. Falls back to the older force-relaxation loop if the energy optimizer does not converge.

This is still a quasi-static approximation, but it is more accurate and less step-size sensitive than the original force-relaxation loop.

## Time-Domain Energy Evaluation

The preferred evaluator is:

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py
```

By default it runs one randomized batch of generated motion/torque profiles:

```text
generated profiles = 30
duration = 5 seconds
samples = 300 per trajectory
```

The generated profiles vary stiffness, cubic torque terms, motion amplitude, motion frequency, phase, bumps, and smoothed noise. The evaluator reports both whole-batch averages and terrain-family averages.

For each timestep it computes:

```text
residual_torque = tau_target - tau_spring
baseline_motor_power = max(0, tau_target * theta_dot)
motor_power_with_spring = max(0, residual_torque * theta_dot)
```

Then it integrates motor power over time:

```text
baseline_motor_energy = integral(baseline_motor_power dt)
motor_energy_with_spring = integral(motor_power_with_spring dt)
energy_saved = baseline_motor_energy - motor_energy_with_spring
offload_percent = 100 * energy_saved / baseline_motor_energy
```

This answers: across a broad set of generated movement trajectories, how much positive motor energy does the spring network remove on average?

## Latest Trajectory Results

Latest results are saved in:

```text
tables/trajectory_efficiency_summary.csv
```

These results use internal-node energy minimization and the current motion-window adaptive trained model. The current default batch has 30 generated profiles.

| Group | Cases | Average offload % |
|---|---:|---:|
| `overall` | 30 | 72.62 |
| `flat_terrain` | 10 | 76.78 |
| `mixed_terrain` | 10 | 74.22 |
| `rough_terrain` | 10 | 66.86 |

```text
adaptive_trained_model overall average offload = 72.62%
average energy saved = 96.331 J
average mean absolute torque error = 10.697 N*m
```

The direct trajectory comparison against the fixed baseline is saved in:

```text
tables/trajectory_model_comparison.csv
```

| Group | Baseline offload % | Adaptive offload % |
|---|---:|---:|
| `overall` | 61.17 | 72.62 |
| `flat_terrain` | 63.79 | 76.78 |
| `mixed_terrain` | 66.09 | 74.22 |
| `rough_terrain` | 53.62 | 66.86 |

The adaptive model now wins overall and across all three terrain families on the current generated evaluation batch.

The window-size sweep is saved in:

```text
tables/window_size_sweep_summary.csv
```

This was the earlier sweep used to choose the 10-sample motion window. The active model has since been retrained with the relaxed-node training/evaluation match, so use the latest trajectory results above for current performance numbers.

| Window | Overall offload % | Flat % | Mixed % | Rough % |
|---:|---:|---:|---:|---:|
| 5 | 61.86 | 77.15 | 57.75 | 50.68 |
| 10 | 67.83 | 85.95 | 64.33 | 53.23 |
| 15 | 57.95 | 74.92 | 51.53 | 47.41 |
| 20 | 53.68 | 66.32 | 48.65 | 46.07 |

Window size 10 is the best tested setting and is now the active model.

## Useful Commands

These commands assume you run them from the repository root.

If Matplotlib warns that it cannot write to its config directory, prefix plotting commands with `MPLCONFIGDIR=/private/tmp`. If you are running on a headless machine and do not want plot windows, also add `MPLBACKEND=Agg`.

### Inspect The Baseline Model

Use this when you want a quick visual and numerical sanity check of the physical spring network.

```bash
python spring-network/02_baseline_profiles/main.py
```

This loads `topologies/baseline_model.json`, evaluates the network at a few joint angles, prints spring stretch and torque values, and saves:

```text
plots/demos/spring_network_demo.png
```

To inspect a different topology file:

```bash
python spring-network/02_baseline_profiles/main.py --topology path/to/topology.json
```

### Plot Baseline Torque Profiles

Use this when you want to see how the fixed `baseline_model` torque curve compares to target terrain torque curves over joint angle.

```bash
python spring-network/02_baseline_profiles/evaluate_profiles.py
```

This prints spring stiffnesses, computes the baseline torque curve, compares it against `flat_terrain`, `rough_terrain`, and `mixed_terrain`, reports quasi-static offload for each profile plus average offload, and saves a plot under:

```text
plots/torque_profiles/
```

The adaptive trained model is not evaluated here because it needs real motion-window history. Use the trajectory comparison command below for that.

### Compare Baseline And Adaptive Models

Use this when you want a fair model comparison over real generated trajectories. This feeds the adaptive model actual motion-window inputs from each trajectory.

```bash
python spring-network/05_trajectory_evaluation/compare_networks.py
```

This compares the two kept cases, `baseline_model` and `adaptive_trained_model`, on the same generated trajectory batch. It reports overall averages plus terrain-family averages.

It saves:

```text
tables/trajectory_model_comparison.csv
```

### Batch Evaluate The Motion-Window Adaptive Trained Model

Use this as the main evaluation command for the current trained model.

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py
```

This uses `models/adaptive_trained_model.npz` by default, builds motion-window features from recent `theta`, `theta_dot`, and `theta_ddot`, predicts changing spring stiffnesses over time, evaluates the generated-profile batch, prints whole-set and terrain-family averages, and saves:

```text
tables/trajectory_efficiency_summary.csv
```

### Batch Evaluate The Baseline Model Over Time

Use this when you want average time-domain energy/offload for the fixed baseline spring network across many generated trajectories.

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --baseline
```

This disables the adaptive model and evaluates the fixed-stiffness `baseline_model`.

```text
tables/trajectory_efficiency_summary.csv
```

Useful batch options:

- `--batch-count 60` changes the number of generated profiles.
- `--batch-seed 7` changes the generated profile set.
- `--samples 1000` increases trajectory resolution for every batch case.
- `--duration 8` changes the simulated duration for every batch case.

### Evaluate One Specific Trajectory

Use `--single` when you want the old one-trajectory behavior with a detailed plot and CSV.

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile mixed_terrain
```

Single-trajectory profile choices:

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile flat_terrain
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile rough_terrain
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile mixed_terrain
```

Single-trajectory evaluation writes outputs to:

```text
plots/trajectory_evaluation/  (created when `--single` is run)
```

Useful trajectory options:

- `--amplitude-deg 25` changes the joint-angle amplitude.
- `--frequency-hz 1.5` changes the motion frequency.
- `--trajectory path/to/file.csv` evaluates a custom trajectory CSV instead of the synthetic sinusoid.
- `--output-dir path/to/folder` writes plots and CSVs somewhere else.

### Retrain The Motion-Window Adaptive Trained Model

Use this only when you want to overwrite the current trained model with a newly trained one.

```bash
python spring-network/04_adaptive_learning/train_adaptive_dataset.py --iterations 1500 --learning-rate 0.01
```

This generates many synthetic terrain trajectories, trains the motion-window adaptive stiffness model, and writes:

```text
models/adaptive_trained_model.npz
tables/adaptive_trained_model_train_results.csv
tables/adaptive_trained_model_test_results.csv
tables/adaptive_trained_model_test_torque_trace.csv
plots/dataset_examples/adaptive_trained_model_test_examples.png
```

Training now uses the same relaxed-node mechanics as trajectory evaluation for its torque basis. After the neural network predicts stiffnesses, the reported train/test metrics and torque trace are computed by applying those stiffnesses to the spring network, relaxing internal nodes, and then measuring spring torque.

The torque trace CSV contains one row per held-out test sample. It includes target torque, final predicted spring torque, residual torque, motion state, and the predicted stiffness value for every spring. Normal training uses 30 held-out test profiles with 160 samples each, so this file has 4,800 data rows.

Useful training options:

- `--train-profiles 80` changes the number of generated training trajectories.
- `--test-profiles 24` changes the held-out test set size.
- `--window-size 10` changes how many recent motion samples the model sees. The current active model uses 10.
- `--iterations 1500` controls training length.
- `--learning-rate 0.01` controls the optimizer step size.
- `--seed 7` makes a training run reproducible.

### Optional Research Commands

This script is still available for experiments, but its generated artifacts are not part of the current kept result set.

Use this to create a stiffness-optimized topology for one terrain profile:

```bash
python "spring-network/03_stiffness_optimization (optional)/optimize_stiffness.py" --target rough_terrain --iterations 200 --learning-rate 0.8
```

This keeps the same node and spring connections as `baseline_model`, but changes each spring's stiffness value to better match one selected terrain torque profile. It writes a new topology JSON and a stiffness-optimization plot. Use it only for experiments where you want a new fixed-stiffness comparison topology; it is not needed for the current baseline/adaptive trained model workflow.

## Current Limitations

- Internal-node equilibrium is solved by spring-energy minimization, but it is still a simple quasi-static model.
- Adaptive models change stiffness values directly, which is a learning abstraction rather than a physical actuator design.
- The adaptive trained model does not receive target torque, so it can only infer terrain context from recent motion.
- The neural-network gradient uses a relaxed torque-basis approximation, while final training metrics and trajectory evaluation use the full relaxed network torque calculation.
- The model is still quasi-static and does not simulate inertia, damping, motor limits, or true robot dynamics.

## Likely Next Steps

- Add constraints for maximum spring force, spring stretch, stiffness bounds, and package size.
- Improve the equilibrium solve for internal nodes.
- Use measured or project-specific torque trajectories instead of only synthetic terrain profiles.
