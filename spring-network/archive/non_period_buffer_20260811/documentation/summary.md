# Spring Network Simulator Summary

## July 28, 2026: preliminary 3D topology-search update

The repository now also contains a genuine three-dimensional, quasi-static
spring-network pipeline under `spring-network/01_core_model/mechanics_3d.py`
and `spring-network/04_adaptive_learning/train_adaptive_3d.py`. This work is
newer than the 2D results summarized below and is not yet a completed
paper-quality experiment.

A randomized search evaluated 180 surface-mounted designs with 24, 32, 40,
48, 56, and 64 linear springs. It varied spring count and continuous 3D node
positions while requiring 50 mm minimum spring centerline spacing. No
64-spring candidate passed the quick feasibility screen. The current
best-found dense-valid candidate is:

`spring-network/topologies/spatial/global_search/candidate_0131_56s.json`

It has 56 springs, 14 free internal nodes, 53.63 mm minimum spring spacing,
and no detected spring, limb, or bearing intersections over a 181-angle,
300-step audit. This is the best result found under a finite randomized search,
not proof of a global optimum.

A preliminary 300-iteration screen on 30 held-out profiles reported 47.34%
mean offload, but this is not a final paper result. Torque changed by only
about 0.015-0.018 N*m RMS when final relaxation increased from 800 to
1200-1600 steps, while a few internal force residuals remained high. The
reported torque was stable, but worst-state force convergence and
adaptive-state collision checks remain required.

The 3D mechanics now supports a consistent cubic hardening term in both force
and potential energy. With cubic ratio 0.5 at a 0.6 m reference extension, the
static torque-authority proxy increased by roughly 6-7% on the leading
48/56-spring designs. No matched nonlinear training result exists yet, so
linear springs remain the selected default.

Exact deferred three-seed, 5000-iteration commands are documented in
`spring-network/PAPER_RUNS.md`.

> **Current active fan:** `spring-network/topologies/adaptive_stiffness/internal_fan_20_spring_model.json` with 20 springs. The old 40-spring fan, models, plots, tables, Isaac Lab work, and earlier PEJ implementation are retained under archive or legacy folders, but they are not the current training path.

Last updated: 2026-07-27

## Current purpose

This repository models passive assistance around one two-dimensional revolute
robot joint. It currently compares two causal neural controllers:

- **Adaptive stiffness:** predicts one stiffness for each of 20 springs at
  every timestep.
- **Adaptive preload:** predicts one preload/rest-length command for each of 20
  springs at every timestep while material stiffness remains fixed.

The project asks what torque each network produces, how closely it can imitate
arbitrary piecewise-linear torque-angle demands, and how much residual motor
torque and energy remain. For preload, it also estimates how much apparent
motor saving is consumed by changing preload.

This is a compact, quasi-static mechanical study. It is not yet a complete
robot, thermal, motor, or hardware simulation.

## Current validated results

The current defensible stiffness results use 300 internal-node relaxation
steps during operating-point refreshes and final evaluation. The held-out set
contains 1,200 profiles.

| Controller | Mean RMSE | Mean motor offload | Mean absolute residual | Worst peak residual |
|---|---:|---:|---:|---:|
| Linear stiffness, refreshed | 38.760 Nm | 36.482% | 29.062 Nm | 205.61 Nm |
| Cubic stiffness, refreshed | 38.664 Nm | 37.219% | 28.632 Nm | 208.37 Nm |
| Linear preliminary hybrid correction | 39.030 Nm | 35.064% | 29.749 Nm | 206.39 Nm |

The cubic run used `c=0.5` with a 600 mm reference extension. It improves mean
offload by only 0.737 percentage points over linear stiffness and slightly
worsens the worst peak residual. It improves fewer than half of individual
profiles, so linear stiffness remains the simpler default.

The mechanics audits established that 30 relaxation steps were seriously
inadequate. For the earlier refreshed cubic model, 30 steps reported about
39.6% offload with a maximum internal force residual near 127 N, while the
converged result was about 17.8%. For the newly trained 300-step checkpoints,
moving from 300 to 500 audit steps changes RMSE by about 0.03 Nm and offload by
about 0.06-0.07 percentage points. Torque metrics are therefore stable even
though a strict 0.1 N worst-sample force tolerance is not always reached.

The current 300-step net-energy preload result is:

| Metric | Held-out result |
|---|---:|
| Torque RMSE | 38.771 Nm |
| Gross motor offload | 37.026% |
| Baseline motor energy | 687.36 J |
| Residual motor energy | 432.86 J |
| Gross motor energy saved | 254.50 J |
| Positive preload-adjustment work | 68.70 J |
| Net energy saved | 185.80 J |
| Net energy saving | 27.03% |

Preload train and test net savings are 192.55 J and 185.80 J, respectively,
which indicates reasonable generalization. This is not yet a fair system-level
comparison with adaptive stiffness because preload adjustment work is charged
while stiffness-adjustment energy is not modeled.

The optional preliminary differentiable-mechanics correction used three
phases, eight complete trajectories per phase, ten updates, and 80 unrolled
relaxation steps. It reduced held-out offload from 36.482% to 35.064%, so it is
retained as experimental infrastructure rather than a preferred result.

## Repository orientation

The active work is under `spring-network/`:

- `01_core_model/` - geometry, spring physics, topology loading, equilibrium,
  torque calculation, and visualization.
- `02_baseline_profiles/` - baseline topology and torque-angle inspection.
- `04_adaptive_learning/` - profile generation, periodicity classification,
  adaptive-stiffness training, adaptive-preload training, and preload
  checkpoint evaluation.
- `05_trajectory_evaluation/` - multi-trajectory evaluation and comparison.
- `topologies/adaptive_stiffness/` and `topologies/preload/` - active topology
  files.
- `models/`, `plots/`, and `tables/` - separated into adaptive stiffness,
  preload, Isaac Lab, and legacy results.
- `archive/isaaclab/` - previous Unitree Go2 workspace and rollout tools. Isaac
  Lab is no longer the active training method.
- `archive/legacy/old-approach/` - the earlier broader PEJ/cam code.
- `archive/experiments/` - prior periodicity, spring-count, and update-limit
  tests.
- `archive/preload/` - earlier preload experiments and visualizers.

The old optional stiffness optimizer is archived under
`archive/legacy/03_stiffness_optimization (optional)/`.

## Mechanical model

The simulator contains one joint at the origin and two rigid limbs. Limb 1
points left and is fixed; limb 2 rotates through angle `theta`. Fixed,
limb-attached, and internal nodes surround the joint. Springs connect node
pairs, and forces on limb-2 nodes generate torque about the origin.

For a linear spring:

```text
extension x = current length - rest length
force F = k x
energy U = 0.5 k x^2
```

Internal nodes can relax toward quasi-static equilibrium before torque is
evaluated. The SciPy path uses L-BFGS-B with a force-relaxation fallback. The
batched PyTorch final-mechanics path uses iterative optimization. Inertia,
damping, backlash, friction, motor limits, and multi-joint dynamics are absent.

## Active and legacy topologies

### Baseline

`spring-network/topologies/adaptive_stiffness/baseline_model.json`

```text
4 fixed anchors + 2 limb-1 nodes + 2 limb-2 nodes + 2 internal nodes
10 nodes, 16 springs
```

It is intentionally asymmetric and hand-authored for debugging. The core
loader retains it as the low-level default, while training selects an explicit
preset or path.

### Current 20-spring internal fan

`spring-network/topologies/adaptive_stiffness/internal_fan_20_spring_model.json`

This is the active adaptive-stiffness fan and the basis of the active preload
variant. Spring-count tests found that 20 springs retained useful capacity at
lower complexity than the 40-spring fan, so the project switched permanently
to 20 for current work.

### Current preload topology

`spring-network/topologies/preload/preload_fan_soft_015_long150.json`

This self-contained override derives from the 20-spring fan and currently uses:

```text
stiffness scale = 0.15
rest-length scale = 1.50
```

It was selected after tests with neutral preload, longer and softer springs,
minimum rest lengths, and preload-specific layouts.

### Legacy 40-spring fan

`spring-network/topologies/legacy/internal_fan_model.json`

```text
10 fixed anchors + 2 limb-1 nodes + 6 limb-2 nodes + 4 internal nodes
22 nodes, 40 springs
```

It formed an anchored four-node internal web with fan connections to multiple
limb-2 moment arms. It is retained only for historical comparison.

## Piecewise-linear torque-angle profiles

The active synthetic target is an arbitrary five-knot curve:

```text
first angle = -45 degrees
last angle  = +45 degrees
three random interior angles
five independently sampled torque values, approximately +/-115 N*m
linear interpolation between knots
```

The range used to be +/-44 degrees and was deliberately changed to +/-45. The
construction is inspired by random point-connected torque-angle curves used in
programmable-joint studies, but this simulator uses a symmetric range.

The shared generator is
`spring-network/04_adaptive_learning/profile_generator.py`.

### Roughness classes

Profiles are ranked using:

```text
35% error from a best-fit line
30% excess total variation beyond a monotonic curve
20% variation among segment slopes
15% fraction of segment-direction reversals
```

The lowest, middle, and highest thirds are named `low_roughness`,
`medium_roughness`, and `high_roughness`. These are relative torque-shape classes,
not roughness labels measured from a robot. Motion parameters do not determine
the class.

### Periodicity experiments

Periodicity measures cycle-to-cycle repeatability: a highly periodic gait
produces similar torque when the same phase and angle recur. It differs from
roughness, which describes the shape of one torque-angle graph.

High- and low-periodicity subsets gave similar results. The synthetic data did
not create a sufficiently large learnability difference, and a controller that
already updates every timestep can compensate. Period-limited updates and
per-cycle upper bounds generally reduced performance by removing needed
authority. These remain archived experiments, not defaults.

## Current trajectories and timesteps

Two motion modes exist:

- `randomized` retains random amplitude, frequency, phase, harmonic content,
  bumps, and smoothed noise.
- `triangular` is the current clean experimental choice and repeatedly sweeps
  from -45 degrees to +45 degrees and back.

With `--motion-mode triangular --fixed-frequency-hz 1.0`, there is no random
amplitude or frequency, no random phase, no harmonic fraction, no irregular
bumps, and no smooth noise.

For 5 seconds and 160 samples:

```text
timestep = 5 / (160 - 1) = approximately 0.03145 seconds
```

A sample is one timestep. A trajectory is the full ordered sequence of angle,
velocity, acceleration, and target torque samples for one generated profile.
The triangular trajectory traverses the entire graph repeatedly; it does not
select an isolated random graph segment.

Velocity and acceleration are calculated causally with backward differences.
No centered difference reveals a future sample. Flexing, extending, and
direction reversal are already represented by velocity and acceleration.

## Neural-network inputs and causality

Current default models do **not** receive the complete target profile.
`--include-profile-descriptor` remains for explicit legacy ablations but
defaults to false.

For a ten-sample window, the input is:

```text
10 theta + 10 theta_dot + 10 theta_ddot
10 previous target torque + 10 previous spring torque
+ 10 previous residual motor torque
= 60 inputs
```

At timestep `t`, the MLP sees current/past motion and past realized torque. It
predicts the current command before current torques enter history. Initial
torque history is zero. It receives no roughness class, roughness score, future
motion, future target torque, or complete future torque-angle curve.

The current MLP has one `tanh` hidden layer, normally 256 units. Adaptive
stiffness uses sigmoid-bounded stiffness outputs. Adaptive preload uses bounded
preload outputs around a neutral operating point. Both emit 20 values.

PyTorch records the forward graph. `loss.backward()` applies the chain rule
through torque error, spring torque, output bounds, hidden activation, weights,
and biases. Adam is usual; the stiffness trainer also supports plain SGD.

## Current loss and energy accounting

Selected current experiments use torque MSE only:

```text
loss = mean((spring torque - target torque)^2)
```

For adaptive stiffness, reproduce this with:

```text
--stiffness-weight 0
--stiffness-change-weight 0
--energy-weight 0
```

The stiffness parser still has a historical nonzero default energy weight, so
explicit zero arguments matter. Preload supports torque-MSE, net-energy, and
hybrid objectives. Net-energy training minimizes residual motor electrical
energy plus positive ideal preload-adjustment work; released spring energy
receives zero credit, matching the final ledger.

For torque-MSE stiffness runs, energy is diagnostic. For preload net-energy
runs, residual motor energy and positive preload work directly affect
backpropagation. Signed shaft power is:

```text
residual torque = target torque - spring torque
shaft power = residual torque * theta_dot
positive power = max(shaft power, 0)
braking power = max(-shaft power, 0)
```

The general electrical model supports motoring efficiency and regenerative
efficiency. Recent mechanical-only comparisons use 1.0 motoring efficiency and
0.0 regeneration. Historical 85% motoring/60% regeneration numbers are not
directly comparable.

Preload telemetry separates baseline motor energy, residual motor energy,
gross motor saving, ideal positive preload-adjustment work, and net saving.
No clutch is used in current preload work; clutch experiments were removed so
continuously adaptive preload could be studied directly.

## Adaptive stiffness

The stiffness MLP predicts 20 bounded stiffnesses every timestep. Training uses
a precomputed relaxed spring torque basis:

```text
predicted torque = sum(precomputed basis_i(theta) * predicted k_i)
```

Final evaluation installs all predicted stiffnesses in the topology, relaxes
internal nodes, and recomputes torque. Training gradients therefore use a
locally fixed-basis approximation while reported final metrics use relaxed
mechanics. `--surrogate-refreshes N` reduces this mismatch by dividing training
into `N + 1` phases. Between phases it causally replays the current controller
through relaxed mechanics and rebuilds the per-spring torque basis at those
actual operating points. The refreshed basis exactly matches relaxed torque at
the schedule where it was constructed while remaining inexpensive and
differentiable during the following optimization phase.

For preliminary experiments, optional differentiable-mechanics correction
phases can provide a small number of direct gradients through unrolled
internal-node relaxation:

```text
--mechanics-correction-phases 3
--mechanics-correction-profiles 8
--mechanics-correction-updates 10
--mechanics-correction-relaxation-steps 80
```

Each correction uses complete causal trajectories, accumulates gradients
through relaxed torque one timestep at a time to bound memory use, and then
passes the corrected controller into the next surrogate refresh. This is a
preliminary hybrid method, not the eventual paper-grade configuration that
would differentiate accurate equilibrium mechanics during every optimizer
iteration over the full training regime.

## Adaptive preload

Preload changes commanded rest length:

```text
commanded rest length = nominal rest length - preload
```

The neutral operating point lets the controller increase or decrease preload
without commanding a negative value. Commands maintain nonnegative rest length
and normally enforce a minimum of 5 mm.

Fully differentiable nonlinear mechanics was briefly used inside every
training timestep. It has now been removed. Current preload training uses a
causal local preload surrogate:

```text
spring torque = neutral torque
              + preload sensitivity * (preload - neutral preload)
```

The initial neutral torque and sensitivity curves are precomputed over angle.
With `--surrogate-refreshes N`, training is divided into phases. Relaxed torque,
local per-spring preload sensitivity, and spring lengths for actuator-work
accounting are rebuilt around the controller's actual schedule between phases.
The refresh includes cubic tangent stiffness when cubic springs are enabled.
Full relaxed mechanics remain the independent final train/test evaluation.

## Cubic hardening and softening

Optional cubic elasticity is:

```text
F = k x [1 + c (x / x_ref)^2]
U = 0.5 k x^2 + 0.25 (k c / x_ref^2) x^4
dF/dx = k [1 + 3 c (x / x_ref)^2]
```

- `c = 0`: linear.
- `c > 0`: hardening.
- `c < 0`: softening.

Softening is safety-bounded. A negative coefficient is rejected if tangent
stiffness falls below a configured positive fraction at the design extension.
Arguments are `--cubic-ratio`, `--cubic-reference-extension-mm`,
`--cubic-design-extension-mm`, and `--cubic-min-tangent-ratio`.

A mild starting point is `c=-0.05`, reference extension 600 mm, design
extension 1000 mm, and minimum tangent ratio 0.05. It retains about 58% of
linear tangent stiffness at 1000 mm.

### Findings

An initial `c=1` at a 50 mm reference was badly scaled because some springs
approach about 900 mm extension. A scaled quick test used `c=0.5` and 600 mm.

- Uniform hardening worsened adaptive-stiffness held-out fit and offload.
- Cubic preload produced only a tiny inconclusive held-out improvement and
  worse training behavior.
- Softening stiffness could show excellent training surrogate offload but poor
  final relaxed offload.

The last discrepancy occurs because training independently scales stored
spring curves, while final mechanics changes all stiffnesses and re-relaxes
the nodes. Relaxation changes extension, direction, moment arm, and coupling.
Softening permits more node movement, making the fixed basis especially
inaccurate. Judge cubic experiments by final relaxed metrics, not the training
`offload surrogate`.

## Evaluation plots

Current evaluators can process many held-out profiles and plot target, spring,
residual motor, and combined spring-plus-motor torque versus both time and
angle. In torque-angle panels, faint raw samples and bold angle-bin mean curves
make spring and motor contributions distinct. Combined torque equals target by
construction because residual motor torque is target minus spring torque.

Adaptive-preload torque-angle points can loop or scatter because preload varies
with time: torque is no longer a single-valued function of angle alone.

## Active artifacts

Important current candidates include:

```text
spring-network/models/adaptive_stiffness/stiffness_causal_torque_mse.npz
spring-network/models/adaptive_stiffness/adaptive_stiffness_optimal.npz
spring-network/models/preload/preload_causal_torque_mse.pt
spring-network/models/preload/preload_adaptive_optimal.pt
```

Experimental cubic models remain alongside them. Older topology-specific,
periodicity, spring-count, and motion-only artifacts are in legacy locations or
carry experimental names. Always check stored metadata and output count rather
than relying only on filenames.

## Current commands

From the repository root in PowerShell:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### Train causal adaptive stiffness

```powershell
python spring-network\04_adaptive_learning\train_adaptive_dataset.py `
  --network fan `
  --motion-mode triangular `
  --fixed-frequency-hz 1.0 `
  --profiles-per-family 2000 `
  --test-profiles-per-family 400 `
  --duration 5 `
  --samples 160 `
  --window-size 10 `
  --iterations 5000 `
  --hidden-dim 256 `
  --learning-rate 0.003 `
  --optimizer adam `
  --min-stiffness 1 `
  --max-stiffness 800 `
  --stiffness-update-mode timestep `
  --stiffness-weight 0 `
  --stiffness-change-weight 0 `
  --energy-weight 0 `
  --motoring-efficiency 1.0 `
  --regen-efficiency 0.0 `
  --mechanics-backend torch `
  --mechanics-batch-size 4096 `
  --relaxation-steps 300 `
  --surrogate-refreshes 2 `
  --device cuda `
  --output-name stiffness_linear_refreshed_relax300
```

### Train causal adaptive preload

```powershell
python spring-network\04_adaptive_learning\train_preload_network.py `
  --topology spring-network\topologies\preload\preload_fan_soft_015_long150.json `
  --motion-mode triangular `
  --fixed-frequency-hz 1.0 `
  --profiles-per-family 2000 `
  --test-profiles-per-family 400 `
  --duration 5 `
  --samples 160 `
  --window-size 10 `
  --iterations 5000 `
  --hidden-dim 256 `
  --learning-rate 0.003 `
  --group-mode per-spring `
  --neutral-preload-mm 700 `
  --max-preload-mm 1400 `
  --minimum-rest-length-mm 5 `
  --finite-difference-mm 1 `
  --nonlinear-relaxation-steps 300 `
  --nonlinear-batch-size 4096 `
  --objective net-energy `
  --motor-energy-weight 1.0 `
  --preload-work-weight 1.0 `
  --surrogate-refreshes 2 `
  --motoring-efficiency 1.0 `
  --regen-efficiency 0.0 `
  --device cuda `
  --output-name preload_refreshed_net_energy_relax300
```

The nonlinear options here apply to final evaluation, not training.

### Evaluate adaptive stiffness over many profiles

```powershell
python spring-network\05_trajectory_evaluation\evaluate_trajectory.py `
  --network fan `
  --adaptive-model spring-network\models\adaptive_stiffness\stiffness_causal_torque_mse.npz `
  --motion-mode triangular `
  --fixed-frequency-hz 1.0 `
  --profiles-per-family 100 `
  --duration 5 `
  --samples 160
```

### Audit stiffness mechanics convergence

Evaluate the same causal checkpoint at several relaxation depths, including
cubic mechanics stored in newer checkpoint metadata:

```powershell
python spring-network\04_adaptive_learning\audit_stiffness_checkpoint.py `
  spring-network\models\adaptive_stiffness\stiffness_causal_refreshed.npz `
  --depths 30 80 160 300 `
  --profiles-per-family 10 `
  --device cuda
```

The audit writes a CSV and convergence plot under
`spring-network/tables/mechanics_audits/`. It reports RMSE, offload, torque
disagreement relative to the deepest solve, and the maximum unbalanced force
on any internal node. Use `--profiles-per-family 0` only for a complete and
potentially expensive held-out-set audit. Older cubic checkpoints that predate
stored cubic metadata require explicit `--cubic-ratio` and
`--cubic-reference-extension-mm` arguments.

### Evaluate adaptive preload

```powershell
python spring-network\04_adaptive_learning\evaluate_preload_checkpoint.py `
  spring-network\models\preload\preload_causal_torque_mse.pt `
  --device cuda `
  --example-profiles 6
```

Plot rendering is CPU work. If the Isaac Lab Conda environment produces
NumPy/Matplotlib DLL conflicts, use the working system Python for plotting and
CPU evaluation.

## Codex collaboration recap, July 26-27

The recent Codex session progressed through these decisions and findings:

1. The repository was reorganized around the active 20-spring stiffness and
   preload pipelines, with Isaac Lab, the 40-spring fan, and older PEJ work
   retained as archive or legacy material.
2. Strictly causal inputs were confirmed: motion windows and previously
   realized torque are available, but the future torque curve and roughness
   class are not.
3. The large difference between the stiffness training surrogate and final
   relaxed mechanics was traced to internal nodes moving when predicted
   stiffness changed.
4. Operating-point surrogate refreshes were implemented. A refresh causally
   replays the controller, relaxes internal nodes, and rebuilds per-spring
   torque contributions around the actual predicted schedule.
5. Initial 30-step refreshed runs appeared to improve offload substantially,
   but a new mechanics audit showed that 30-step relaxation was far from
   equilibrium and overstated performance.
6. The audit utility was added to compare checkpoints across relaxation
   depths, measure torque stability, and report mean and maximum unbalanced
   force on internal nodes.
7. Matched 300-step linear and cubic stiffness models were trained. Cubic
   hardening produced only a small average advantage, so linear remains the
   primary baseline.
8. Smoothing was investigated. It was judged inappropriate as a primary
   theoretical constraint because the present study seeks unconstrained upper
   bounds. Smooth commands also do not guarantee smooth torque.
9. Redundant `spring + motor` curves were removed from torque-angle panels.
   That curve always equals the target because residual motor torque is defined
   as `target - spring torque`.
10. Preload training was changed from hard-coded torque MSE to explicit
    torque-MSE, net-energy, and hybrid objectives. Net-energy backpropagation
    now charges residual motor electrical energy plus positive preload work.
11. Preload operating-point refreshes were added using relaxed torque,
    cubic-aware local preload sensitivity, and actual spring lengths for work
    accounting.
12. A preliminary stiffness mode was added that backpropagates through
    unrolled relaxed mechanics during a few small correction phases. Its first
    matched experiment reduced held-out performance, so it is not the current
    recommended model.
13. The current paper-facing interpretation is deliberately conservative:
    300-step relaxed mechanics, linear stiffness as the default, cubic as a
    small-effect ablation, preload reported with both gross and net energy, and
    older 30-step results treated as relaxation artifacts.
14. A genuine spatial mechanics branch was added. Unlike the rejected layered
    illustration, it uses real `[x, y, z]` node coordinates, relaxes every
    internal node in all three coordinates, and projects `r x F` onto the
    configured revolute-joint axis.

## Preliminary genuine-3D experiment (2026-07-27)

The first spatial experiment uses
`topologies/spatial/internal_fan_3d_28_spring.json`: 28 nodes, 28 springs, 12
fixed anchors distributed around a cylindrical joint housing, 6 unconstrained
3D internal nodes, 4 limb-1 nodes, and 6 rotating limb-2 nodes. The joint is
still a single revolute degree of freedom (now configured about y); “3D” refers to the spring
geometry, internal equilibrium, and force calculation rather than a
three-degree-of-freedom joint.

The matched exploratory run used 6,000 training profiles, 1,200 held-out
profiles, 160 samples per profile, a ten-sample causal history, two
operating-point refreshes, 160 relaxation steps per spatial solve, and 5,000
neural optimizer iterations on CUDA. Runtime was 1,306 seconds.

| Model | Held-out mean RMSE | Held-out median RMSE | Mean motor offload |
|---|---:|---:|---:|
| fixed spatial baseline | 61.811 N*m | 61.720 N*m | -20.230% |
| adaptive spatial | 15.625 N*m | 14.207 N*m | 81.123% |

Training and test means were close (15.680 versus 15.625 N*m), which is
evidence of good generalization within this synthetic distribution. The large
gain over the fixed spatial baseline is not evidence that 3D is inherently
better than the planar model: this is a new 28-spring geometry with different
preload and torque authority, so a topology-matched planar ablation is still
required.

The baseline 160-step solve reached a maximum internal force residual of
0.01089 N. Under learned stiffness schedules, final residuals were 0.101 N
mean and 2.061 N maximum on held-out data. The torque results are promising
but preliminary; paper-grade reporting should repeat or audit final evaluation
at 300 or more relaxation steps, especially for the worst learned
high-stiffness states.

Spatial artifacts are stored in `models/spatial/`, `plots/spatial/`, and
`tables/spatial/`. The cleaned topology render is
`plots/spatial/internal_fan_3d_28_spring_topology.png`.

### Feasible y-axis topology screening

The initial cylindrical/z-axis concept was replaced with a y-axis revolute
joint, tapered 3D limbs, a shortened bearing, and separate front/back surface
attachments. A GPU feasibility audit now rejects unused nodes, disconnected
graphs, fixed-to-fixed springs, free internal nodes with fewer than three
springs, unsupported limb attachments, and relaxed spring centerlines that
intersect either limb or the bearing over -45 to +45 degrees.

An early collision-barrier run produced 75.60% held-out offload but had roughly
60 N mean equilibrium residual and is invalid for mechanics comparison.
Re-routing the two offending springs removed the need for the stiff barrier.
The corrected 30-spring baseline reaches 0.0046 N mean residual at 160 steps
and approximately 0.000009 N at 300 steps on the baseline angle sweep.

Matched preliminary CUDA screens used 3,000 training profiles, 600 held-out
profiles, 2,000 optimizer iterations, one mechanics refresh, and 160
relaxation steps:

| Springs | Held-out RMSE | Held-out offload | Mean/max force residual |
|---:|---:|---:|---:|
| 24 | 34.072 N*m | 44.661% | 0.0589 / 2.528 N |
| 30 | 19.540 N*m | 74.032% | 0.0441 / 1.997 N |
| 36 | **16.436 N*m** | **79.440%** | **0.0255 / 1.533 N** |

That first screen made 36 springs the best configuration tested at that stage,
but did not establish an optimum. More spring counts, attachment placements,
seeds, and deeper learned-state evaluation remained necessary.

A second, smaller screen added true top/bottom edge attachments while
preserving the front/back offset required to avoid the rotating limb. It used
1,500 training profiles, 300 held-out profiles, 1,000 iterations, one refresh,
and otherwise matched CUDA mechanics:

| Springs | Held-out RMSE | Held-out offload | Mean/max force residual |
|---:|---:|---:|---:|
| 36 | 16.243 N*m | 77.726% | 0.0251 / 1.312 N |
| 42 | 16.015 N*m | 78.122% | 0.0259 / 2.524 N |
| 48 | **15.850 N*m** | **79.422%** | 0.0262 / 0.966 N |

Performance was still increasing at 48 springs, but the gains were much
smaller than the 24-to-30 spring gain. Therefore 48 is the best tested
topology, not a demonstrated optimum. The fixed baseline also becomes more
overpowered as springs are added, so adaptive offload, package complexity,
force limits, and stiffness-actuation cost must eventually be optimized
together rather than maximizing spring count alone.

A matched 54-spring continuation reduced held-out offload to 78.623% and
increased RMSE to 15.948 N*m. The observed small-screen pattern therefore
peaked at 48 springs among 36, 42, 48, and 54. The planned 60-spring run was
not needed for the practical selection rule: once extra springs reversed the
gain, 48 was retained as the simpler candidate.

### Deep-relaxation preload spot check

A deliberately small planar preload run used 150 training profiles, 30
held-out profiles, 200 controller iterations, one refresh, a net-energy
objective, and 300 nonlinear relaxation steps on CUDA. It produced -18.89%
held-out gross motor offload and -101.99 J net energy saving (that is, a
101.99 J net penalty), with only 0.16% RMSE improvement. This is evidence that
the present 2D preload configuration is not beneficial under this matched
deep-relaxation setup. It is not a 3D preload result: the preload trainer still
loads planar topology data and hard-codes planar torque.

## Historical results retained for context

These came from the older 40-spring, profile-aware,
positive-mechanical-work pipeline. They predate the present 20-spring strictly
causal model, triangular motion, and current energy settings and are not
directly comparable.

Historical trajectory summary:

| Group | Cases | Average offload |
|---|---:|---:|
| overall | 30 | 72.62% |
| low_roughness | 10 | 76.78% |
| medium_roughness | 10 | 74.22% |
| high_roughness | 10 | 66.86% |

Historical fixed-versus-adaptive comparison:

| Group | Fixed | Adaptive |
|---|---:|---:|
| overall | 61.17% | 72.62% |
| low_roughness | 63.79% | 76.78% |
| medium_roughness | 66.09% | 74.22% |
| high_roughness | 53.62% | 66.86% |

Historical window sweep:

| Window | Overall | Flat | Mixed | Rough |
|---:|---:|---:|---:|---:|
| 5 | 61.86% | 77.15% | 57.75% | 50.68% |
| 10 | 67.83% | 85.95% | 64.33% | 53.23% |
| 15 | 57.95% | 74.92% | 51.53% | 47.41% |
| 20 | 53.68% | 66.32% | 48.65% | 46.07% |

The files are retained under `spring-network/tables/legacy/` and
`spring-network/plots/legacy/`. They support the historical ten-sample-window
choice but require revalidation under the current pipeline.

## Current limitations

- Single-joint, quasi-static mechanics only. A spatial spring branch now
  exists, but the joint itself remains one revolute degree of freedom.
- Roughness-class names describe thirds of the synthetic torque-shape ranking.
- Strict causality prevents anticipating unrelated future torque demands.
- Adaptive stiffness/preload are abstractions rather than complete actuators.
- Most stiffness optimizer updates use a refreshed local torque basis rather
  than differentiating exact equilibrium mechanics at every update.
- Preload uses a refreshed local sensitivity model between full relaxed
  operating-point reconstructions.
- The preliminary unrolled stiffness correction uses a small subset and fewer
  relaxation steps than final mechanics and did not improve held-out results.
- Uniform cubic coefficients poorly match different spring extension ranges.
- Preload energy is an ideal lower bound without real transmission losses.
- Stiffness-adjustment actuator energy is not yet modeled, so stiffness motor
  offload and preload net saving are not system-level equivalents.
- Worst-sample internal force residual can remain above 0.1 N even after torque
  and offload have numerically stabilized.
- Filenames alone do not establish model/topology/preprocessing compatibility.

## Likely next steps

1. Add a preload mechanics-convergence audit analogous to the stiffness audit.
2. Run complete held-out stiffness audits at 160, 300, and 500 steps when
   publication-level compute is available.
3. For paper-grade differentiable training, replace the preliminary correction
   subset with exact or implicit equilibrium gradients at every optimizer
   update and validate them against finite differences.
4. Model stiffness-adjustment energy so stiffness and preload can be compared
   on the same net-system-energy basis.
5. Test bounded per-spring cubic coefficients normalized by each spring's
   observed extension range only if the added complexity is justified.
6. Add force, extension, package-size, actuator-work, and eventually physical
   rate limits after the unconstrained upper-bound study is complete.
7. Validate causal models on measured or project-specific torque trajectories.
8. Revisit robot rollouts after surrogate, final mechanics, and actuator energy
   agree reliably.
9. Audit the learned spatial controller at 300 and 500 relaxation steps and
   build a topology-matched planar ablation before making a 2D-versus-3D claim.

## Paper-figure and table status (updated 2026-07-28)

The curated paper assets are under
`spring-network/plots/paper_figures/`. Figure 1 shows the selected
collision-audited candidate 131 topology with 56 linear springs at +25
degrees. Its original geometry and camera orientation are preserved; only the
displayed coordinate names are assigned so X is left-right, Y is vertical,
and Z is forward-back.

Figure 3 is split into two files:

- `fig03a_torque_angle_profiles.png`
- `fig03b_torque_time_profiles.png`

The three examples are drawn from the deterministically regenerated held-out
dataset rather than hand-authored curves. Terrain and roughness categories are
not displayed. The target uses the same black dashed style as the test
reports. The torque-time figure uses 0.25 Hz for 20 seconds, giving five
complete cycles with 640 samples.

Figure 4 is one reconstructed held-out test from the saved 48-spring
checkpoint. Its middle panel shows the piecewise-linear target and relaxed
spring torque versus joint angle.
The stiffness heatmap rows are ordered by 3D proximity between neutral-position
spring midpoints; the row number is therefore not the topology-file Spring ID.
This ordering changes only the visualization, not the controller or mechanics.
Collective stiffness transitions are genuine controller outputs: the shared
MLP updates all springs each timestep, and no stiffness-rate or
stiffness-change constraint is applied. All three panels use the same causal
rollout, reconstructed using the checkpoint's saved seed, profile split,
normalization, 1 Hz motion, duration, sample count, and test-dataset seed.

Publication-table candidates are under
`spring-network/plots/paper_figures/tables/` in both PNG and CSV form:

- `table01_primary_performance_preliminary` compares fixed and adaptive
  mechanics for the short candidate-131 screen.
- `table02_selected_topology_feasibility` reports the dense geometric audit,
  including 53.63 mm minimum spring clearance against the 50 mm requirement
  and zero detected spring-spring, limb, or bearing violations.
- `table03_mechanical_convergence` reports relaxation-depth sensitivity for
  the preserved 2D linear and cubic mechanics audits.

An earlier spring-count table showing 70-79% offload was removed from the
paper assets. Those numbers came from older 1,000-iteration runs with 1,500
training profiles, 300 held-out profiles, 160-step evaluation, and different
topologies. They are genuine historical outputs but are not comparable to the
current 56-spring candidate-131 screen, which used only 300 iterations, 90
training profiles, 30 held-out profiles, and 800-step final evaluation.

The multiple-training-seed robustness figure is implemented but cannot yet be
populated honestly. Only seed 202 currently exists for candidate 131; other
saved seeds correspond to different topologies. The matched 5,000-iteration
GPU runs for seeds 401, 503, and 607 are specified in
`spring-network/PAPER_RUNS.md`. When their three mechanics-comparison files
exist, `generate_paper_figures.py` will create
`fig06_multiple_seed_robustness.png` automatically.
