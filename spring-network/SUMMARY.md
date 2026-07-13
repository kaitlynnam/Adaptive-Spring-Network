# Spring Network Simulator Summary

This folder contains the active spring-network approach for modeling passive assistance around a single 2D revolute robot joint.

## Repository Orientation

The repository contains three related work areas:

- `spring-network/` is the active, compact single-joint spring-network simulator described in this document.
- `isaaclab-go2/` is a small Isaac Lab workspace that spawns a Unitree Go2 and is intended to become the source of real robot rollouts.
- `old-approach/` is the earlier, broader Parallel Elastic Joint research workspace. It contains reusable PEJ equations, simulation examples, cam-network studies, and tests, but it is not the active spring-network implementation.

The intended data flow is:

```text
generated profiles or Isaac Lab rollout
    -> one-joint trajectory (theta, theta_dot, tau_target)
    -> spring-network topology and stiffness model
    -> relaxed spring torque
    -> residual motor torque, bidirectional motor energy, and energy offload
```

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

- `06_isaaclab_export/`
  Exports Go2 rollouts from an Isaac Lab/RSL-RL checkpoint and converts one selected joint into the CSV schema accepted by the trajectory evaluator.

## Current Topologies And Models

Topology files live in `topologies/`. Adaptive model files live in `models/`.

There are currently two topology JSON files. A topology defines node types, initial node positions, spring connections, and initial spring stiffnesses. It is separate from the choice between fixed and neural/adaptive stiffness.

### `topologies/baseline_model.json`

This is the hand-picked baseline structure. It has:

- 10 nodes
- 16 springs
- fixed anchors
- limb-1 nodes
- limb-2 nodes
- internal nodes

The structure is intentionally asymmetric and manually chosen so the model is easy to inspect and debug. It is not a perfect grid or optimized web.

Node and connection counts:

```text
4 fixed anchors + 2 limb-1 nodes + 2 limb-2 nodes + 2 internal nodes = 10 nodes
16 springs
```

This remains the low-level loader default because `01_core_model/topology_loader.py` sets `DEFAULT_TOPOLOGY_PATH` to this file. User-facing training and trajectory evaluation now use explicit network presets instead.

### `topologies/internal_fan_model.json`

This is the larger, higher-authority experimental web. It adds more anchors, four relaxable internal nodes, and six attachment points on rotating limb 2.

```text
10 fixed anchors + 2 limb-1 nodes + 6 limb-2 nodes + 4 internal nodes = 22 nodes
40 springs
```

Its springs form an anchored internal four-node web plus fan-like connections to several limb-2 moment arms. It is the default `fan` preset for user-facing training and trajectory evaluation.

### Untuned fixed-stiffness initialization

The node positions and spring connections remain hand-authored, but the initial `stiffness_k` values are deliberately untuned reproducible random controls:

| Topology | Distribution | Range | Seed | Mean sampled stiffness |
|---|---|---:|---:|---:|
| `baseline_model.json` | log-uniform | 5–150 N/m | 101 | 58.675 N/m |
| `internal_fan_model.json` | log-uniform | 5–150 N/m | 202 | 41.754 N/m |

The fixed seeds make experiments reproducible while avoiding the earlier hand-picked stiffness values that happened to align with the synthetic target distribution. Randomization does not normalize aggregate torque authority: the fan still has more springs and can remain stronger or weaker depending on geometry. Compare fixed and adaptive behavior within each topology before comparing topologies.

The currently kept trained model files were produced before this stiffness randomization. Retrain both models so their initialization penalty and training mechanics match the new topology JSON files.

### Trained model artifacts

The `models/` directory intentionally contains only the two topology-specific trained counterparts:

```text
adaptive_trained_baseline_model.npz
adaptive_trained_internal_fan_model.npz
```

Both use the current 60-input `motion_torque_window` format, a 10-sample window, 256 hidden units, PyTorch mechanics, CPU training, and `energy_weight=0.35`. Each was trained as a small model on 100 profiles per terrain family and tested on 30 held-out profiles per family, with 160 samples per trajectory.

The baseline model emits 16 stiffnesses for the 16-spring baseline topology. The internal-fan model emits 40 stiffnesses for the 40-spring fan topology. Historical, window-sweep, generic, and smoke model files have been removed from `models/`.

### Topology/model compatibility

| Topology | Springs | Fixed stiffness | Trained adaptive counterpart |
|---|---:|---|---|
| `baseline_model.json` | 16 | Compatible | `adaptive_trained_baseline_model.npz` (16 outputs) |
| `internal_fan_model.json` | 40 | Compatible | `adaptive_trained_internal_fan_model.npz` (40 outputs) |

The evaluator validates this count before applying stiffnesses and raises an error if a custom topology and model do not match.

## Piecewise-Linear Torque Profiles

The current project represents target torque curves as five-knot piecewise-linear profiles. It supports two profile sets:

- `terrain` first creates one shared population of random motions and five-knot restoring torque curves. It then ranks that population by roughness and labels equal-sized thirds `flat_terrain`, `mixed_terrain`, and `rough_terrain`. This is the training script's default.
- `arbitrary` samples all five torque knots independently. It remains available for generic profile experiments and single-trajectory evaluation, but the main trajectory batch now uses the ranked terrain set.

The profile generator follows the same graph type described by Wu et al. in the ERC programmability study: target torque responses are generated by connecting random points in rotation-angle versus torque space.

Every generated target uses:

```text
5 torque-angle points, matching the paper's five-point construction
first and last angles fixed at -44 deg and +44 deg for this simulator's joint range
3 interior angles sampled randomly
target torque evaluated with linear interpolation between points
```

For `arbitrary` profiles, all five torque values are sampled independently in +/-115 N*m. For terrain training, a random restoring curve is generated from independently sampled negative-side stiffness, positive-side stiffness, cubic stiffness, and knot noise, then clipped to +/-115 N*m. No terrain label influences generation.

The terrain roughness score combines:

```text
60% motion irregularity:
    frequency, harmonic content, bump count, and noise
40% torque-curve difficulty:
    slope magnitude, slope variation, and torque range
```

After scoring, the smoothest third becomes `flat_terrain`, the middle third becomes `mixed_terrain`, and the roughest third becomes `rough_terrain`. Training and held-out populations are generated and classified independently. These are relative synthetic categories, not terrain labels measured from a robot.

The paper uses the same construction over a 0 deg to 90 deg ERC rotation domain. This simulator uses a symmetric joint-angle domain because the spring-network model is evaluated around zero. The shared implementation is in `04_adaptive_learning/profile_generator.py`. Baseline angle-domain plots, adaptive training, batch trajectory evaluation, and model comparison all use this generator.

New training uses ten causal samples of six signals:

```text
motion history:
    theta, theta_dot, theta_ddot
realized torque history:
    previous controller-demanded target torque
    previous spring torque produced by the selected stiffness
    previous residual motor torque = target torque - spring torque

10 samples * 6 signals = 60 inputs
```

At each timestep the model predicts stiffness before the current torque values are added to history. The mechanics solver then computes realized spring torque, residual motor torque is derived, and those three torque values become available only to the next prediction. The first torque-history window is zero because nothing has happened yet. Complete future profiles and current/future target torque are never model inputs.

Profile-knot inputs have been removed. Evaluation uses the new `motion_torque_window` format for both kept models; `motion_window` parsing remains only for explicitly supplied external historical artifacts.

Newly generated motion features are causal throughout: each window contains only the current and previous samples, and missing velocity/acceleration values are reconstructed with backward differences rather than centered differences. The first reconstructed derivative is initialized to zero because no earlier sample exists.

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

By default it runs one balanced randomized batch using the same rank-classified terrain generator as training:

```text
flat profiles = 100
mixed profiles = 100
rough profiles = 100
total profiles = 300
duration = 5 seconds
samples = 160 per trajectory
```

The five-second duration and 160-sample resolution intentionally match the adaptive training defaults. This keeps a ten-sample causal history at the same approximate physical duration during training and evaluation. If either duration or sample count is changed for adaptive evaluation, retrain at the same timing until the model supports time-based history resampling.

The generated profiles vary piecewise-linear torque knots, motion amplitude, motion frequency, phase, bumps, and smoothed noise. They are ranked by roughness and divided into equal terrain thirds. The evaluator reports whole-batch averages followed by separate `flat_terrain`, `mixed_terrain`, and `rough_terrain` averages for offload, energy saved, mean absolute torque error, and maximum absolute torque error.

### What topology does the evaluator run?

`evaluate_trajectory.py` provides paired `--network baseline` and `--network fan` presets. With no arguments, it runs:

```text
network preset = fan
topology       = topologies/internal_fan_model.json (40 springs)
stiffness mode = adaptive_trained_internal_fan_model.npz
                 (60 torque-history inputs, 40 outputs)
node handling  = quasi-static internal-node relaxation enabled
evaluation     = 300 ranked piecewise-linear terrain trajectories
                 (100 flat, 100 mixed, 100 rough)
```

Switch between the matched adaptive networks with one option:

```bash
# Adaptive 16-spring baseline; requires its topology-specific torque-history model
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --network baseline

# Adaptive 40-spring internal fan (default)
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --network fan

# Fixed-stiffness versions of either topology
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --network baseline --baseline
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --network fan --baseline
```

`--topology` and `--adaptive-model` remain available as expert overrides. A mismatched spring/output count produces an error rather than silently discarding outputs. An automatic preset also errors if its topology-specific model is missing or still uses the old motion-only feature type.

`--no-relax-internal` disables relaxation for a fixed-stiffness run. The adaptive path currently always requests relaxation, so that flag does not disable it when an adaptive model is selected.

For each timestep it first computes signed shaft power:

```text
residual_torque = tau_target - tau_spring
shaft_power = motor_torque * theta_dot
positive_mechanical_power = max(shaft_power, 0)
braking_mechanical_power = max(-shaft_power, 0)
```

The default bidirectional accounting assumes 85% motoring efficiency and 60% regenerative efficiency:

```text
electrical_draw_power = positive_mechanical_power / 0.85
regenerated_power = braking_mechanical_power * 0.60
unrecovered_braking_power = braking_mechanical_power * (1 - 0.60)
net_battery_power = electrical_draw_power - regenerated_power
energy_burden_power = electrical_draw_power + unrecovered_braking_power
```

The primary offload percentage compares integrated `energy_burden_power` with and without springs. Therefore, over-assistance that forces the motor to brake is no longer treated as free: only the configured regenerative fraction receives credit, and the unrecovered fraction counts against offload. Reports also expose net battery energy, total braking energy, and regenerated energy. Net battery energy can be negative on a highly regenerative synthetic cycle, so it is a diagnostic rather than the primary optimization metric.

Use `--motoring-efficiency` and `--regen-efficiency` to match a particular drive. Both training and evaluation default to the same values and reject efficiencies outside their physical ranges.

## Latest Trajectory Results

Latest results are saved in:

```text
tables/trajectory_efficiency_summary.csv
```

Result files must be interpreted together with the topology and model that generated them. `trajectory_efficiency_summary.csv` includes a `topology` column; check it before drawing conclusions.

The historical comparison below predates the current causal torque-history, rank-classified profile pipeline, and bidirectional energy accounting. Keep it as experiment history, not as the current benchmark; its offload percentages used positive mechanical work only and are not directly comparable to new results.

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

The historical adaptive model won overall and across all three old terrain families on that generated evaluation batch. Rerun a topology-matched comparison before comparing the current artifacts.

The window-size sweep is saved in:

```text
tables/window_size_sweep_summary.csv
```

This was the earlier sweep used to choose the 10-sample motion window. It predates the current causal preprocessing and rank-based terrain construction, so treat it as historical evidence for the window choice rather than current performance.

| Window | Overall offload % | Flat % | Mixed % | Rough % |
|---:|---:|---:|---:|---:|
| 5 | 61.86 | 77.15 | 57.75 | 50.68 |
| 10 | 67.83 | 85.95 | 64.33 | 53.23 |
| 15 | 57.95 | 74.92 | 51.53 | 47.41 |
| 20 | 53.68 | 66.32 | 48.65 | 46.07 |

Window size 10 was the best setting in that historical sweep and remains the current training default. It should be revalidated after full causal retraining.

## Useful Commands

These commands assume you run them from the repository root.

If Matplotlib warns that it cannot write to its config directory, prefix plotting commands with `MPLCONFIGDIR=/private/tmp`. If you are running on a headless machine and do not want plot windows, also add `MPLBACKEND=Agg`.

### Inspect The Baseline Model

Use this when you want a quick visual and numerical sanity check of the physical spring network.

```bash
python spring-network/02_baseline_profiles/main.py

Visualize the baseline topology with:

  python spring-network/02_baseline_profiles/main.py \
    --topology spring-network/topologies/baseline_model.json

  Visualize the internal fan with:

  python spring-network/02_baseline_profiles/main.py \
    --topology spring-network/topologies/internal_fan_model.json
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

Use this when you want to see how the fixed `baseline_model` torque curve compares to generated piecewise-linear target torque curves over joint angle.

```bash
python spring-network/02_baseline_profiles/evaluate_profiles.py
```

This prints spring stiffnesses, computes the baseline torque curve, compares it against three seeded random piecewise-linear target profiles, reports quasi-static offload for each profile plus average offload, and saves a plot under:

```text
plots/torque_profiles/
```

The adaptive trained model is not evaluated here because it needs real motion-window history. Use the trajectory comparison command below for that.

### Compare Baseline And Adaptive Models

This script feeds the adaptive model actual motion-window inputs from each trajectory:

```bash
python spring-network/05_trajectory_evaluation/compare_networks.py
```

The default comparison now pairs the fixed case with `baseline_model.json` and the adaptive case with `internal_fan_model.json`. Thus it compares the two complete configurations, not only the effect of adaptive stiffness on identical geometry. Use `--baseline-topology` and `--adaptive-topology` to override them; `--topology` remains a legacy shared override and must be compatible with the adaptive model.

It saves:

```text
tables/trajectory_model_comparison.csv
```

### Batch Evaluate The Motion-Window Adaptive Trained Model

Use this as the main evaluation command for the current trained internal-fan model.

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py
```

This selects the `fan` preset by default and uses its 60-input causal motion-and-realized-torque model. The evaluator predicts stiffness sequentially, evaluates the balanced terrain batch, prints overall plus flat/mixed/rough statistics, and saves:

```text
tables/trajectory_efficiency_summary.csv
```

Use `--network baseline` to switch to the paired baseline topology/model.

### Batch Evaluate The Baseline Model Over Time

Use this when you want average time-domain energy/offload for the fixed baseline spring network across many generated trajectories.

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --network baseline --baseline
```

This disables the adaptive model and evaluates the fixed-stiffness `baseline_model`. This is the matched default baseline case.

```text
tables/trajectory_efficiency_summary.csv
```

Useful batch options:

- `--profiles-per-family 20` evaluates 20 flat, 20 mixed, and 20 rough profiles.
- `--batch-count 60` is a total-count shortcut for the same balanced 20/20/20 batch; it must be divisible by three.
- `--batch-seed 7` changes the generated profile set.
- `--samples 1000` increases trajectory resolution for every batch case. For adaptive models, changing the resolution also changes the physical duration represented by the ten-sample history, so retrain with the same `--duration` and `--samples` values.
- `--duration 8` changes the simulated duration for every batch case.

### Evaluate One Specific Trajectory

Use `--single` when you want the old one-trajectory behavior with a detailed plot and CSV.

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile piecewise_0000
```

Single-trajectory default seeded profile choices:

```bash
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile piecewise_0000
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile piecewise_0001
python spring-network/05_trajectory_evaluation/evaluate_trajectory.py --single --profile piecewise_0002
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

Use this to train a topology-specific adaptive model. Training defaults to the internal fan and saves `models/adaptive_trained_internal_fan_model.npz`:

```bash
python spring-network/04_adaptive_learning/train_adaptive_dataset.py --network fan
```

To train the baseline instead, use `--network baseline`; it saves `models/adaptive_trained_baseline_model.npz`. The evaluator automatically prefers these topology-specific files once they exist.

By default this generates 6,000 training trajectories (2,000 for each of three terrain families) and 1,200 held-out trajectories (400 per family), with piecewise-linear target torque profiles. It then trains the adaptive stiffness model and writes:

```text
models/<topology-specific output name>.npz
tables/<output name>_train_results.csv
tables/<output name>_test_results.csv
tables/<output name>_test_torque_trace.csv
tables/<output name>_mechanics_comparison.csv
plots/dataset_examples/<output name>_test_examples.png
plots/dataset_examples/<output name>_training_convergence.png
```

Training uses relaxed-node mechanics for its torque basis. It trains causally from current/past motion plus previously realized target, spring, and residual motor torque. It receives no current/future target torque, future samples, terrain label, roughness score, or target-profile knots. The default MLP has 256 hidden units and predicts one stiffness per spring in the selected topology, in the 1 to 800 N/m range. Training rolls forward in time so each prediction can use the torque consequences of earlier stiffness decisions. Final train/test metrics replay the model sequentially through the full relaxed mechanics.

The PyTorch training loss combines torque RMSE pressure with an energy/offload-aware term using the same bidirectional energy burden as evaluation. It penalizes electrical motoring draw plus the portion of braking energy that is not regenerated. The convergence plot shows training RMSE, loss, the offload surrogate, marks the best-loss iteration, and overlays the fixed-stiffness baseline train/test RMSE for scale.

PyTorch is now required for training and `--mechanics-backend torch` is the default. This workspace currently has PyTorch 2.13.0 installed. On this Apple Silicon environment, CUDA and MPS are unavailable to the current build, so `--device auto` uses the CPU. If PyTorch is unavailable, training stops with a clear error instead of silently falling back to a path that ignores the energy-aware term. `--mechanics-backend scipy` remains an explicit final-mechanics fallback, while neural optimization still requires PyTorch.

The mechanics comparison CSV reports held-out test metrics for:

```text
fixed baseline, relaxed
fixed baseline, unrelaxed
adaptive model, relaxed
adaptive model, unrelaxed
```

The torque trace CSV contains one row per held-out test sample. It includes target torque, final predicted spring torque, residual torque, motion state, and the predicted stiffness value for every spring. Default terrain training uses 1,200 held-out profiles with 160 samples each, so a complete trace has 192,000 data rows.

Useful training options:

- `--profile-set terrain` uses three terrain families; `--profile-set arbitrary` uses independently random torque knots.
- `--profiles-per-family 2000` and `--test-profiles-per-family 400` control the default terrain data set.
- `--train-profiles 12000` and `--test-profiles 1200` apply when `--profile-set arbitrary` is selected.
- `--window-size 10` changes how many recent motion samples the model sees. The current active model uses 10.
- `--iterations 5000` controls training length and shows the current default.
- `--learning-rate 0.01` controls the optimizer step size.
- `--hidden-dim 256` controls MLP width.
- `--max-stiffness 800` controls the upper stiffness bound in N/m.
- `--energy-weight 0.35` controls how strongly training rewards bidirectional motor-energy reduction.
- `--motoring-efficiency 0.85` converts positive shaft work to electrical draw.
- `--regen-efficiency 0.60` controls how much braking work receives regenerative credit; the unrecovered portion counts against offload.
- `--progress-interval 100` prints progress every 100 profiles during dataset/evaluation loops and every 100 optimizer iterations. Use `0` to suppress these progress updates.
- `--device auto` uses the GPU for neural optimization when CUDA is available. Use `--device cuda` to require GPU or `--device cpu` to force CPU.
- `--mechanics-backend torch` is the default batched mechanics path. Use `--mechanics-backend scipy` explicitly for the original SciPy solver.
- `--mechanics-batch-size 8192` changes how many samples are evaluated per PyTorch mechanics batch.
- `--relaxation-steps 80` changes how many PyTorch optimizer steps are used to relax internal nodes in each mechanics batch.
- `--seed 7` makes a training run reproducible.

### Optional Research Commands

This script is still available for experiments, but its generated artifacts are not part of the current kept result set.

Use this to create a stiffness-optimized topology for one generated piecewise-linear profile:

```bash
python "spring-network/03_stiffness_optimization (optional)/optimize_stiffness.py" --target piecewise_0000 --iterations 200 --learning-rate 0.8
```

This keeps the same node and spring connections as `baseline_model`, but changes each spring's stiffness value to better match one selected piecewise-linear torque profile. It writes a new topology JSON and a stiffness-optimization plot. Use it only for experiments where you want a new fixed-stiffness comparison topology; it is not needed for the current baseline/adaptive trained model workflow.

## Current Limitations

- Internal-node equilibrium is solved by spring-energy minimization, but it is still a simple quasi-static model.
- Adaptive models change stiffness values directly, which is a learning abstraction rather than a physical actuator design.
- Past demanded, realized spring, and residual motor torque reduce the ambiguity of motion-only input, but the model still cannot anticipate an unrelated future torque change. This is intentional causal behavior.
- Existing model artifacts do not store topology identity. Trajectory evaluation does validate that neural output count equals spring count, but equal counts alone cannot prove that a custom model was trained on the selected topology.
- The synthetic `flat`, `mixed`, and `rough` labels are relative roughness thirds of each generated population, not measured physical terrain labels.
- The neural-network gradient uses a relaxed torque-basis approximation, while final training metrics and trajectory evaluation use the full relaxed network torque calculation.
- The model is still quasi-static and does not simulate inertia, damping, motor limits, or true robot dynamics.

## Likely Next Steps

- Add constraints for maximum spring force, spring stretch, stiffness bounds, and package size.
- Improve the equilibrium solve for internal nodes.
- Use measured or project-specific torque trajectories instead of only synthetic piecewise-linear profiles.
