# Spring Network Research Code

## Active pipeline

The active experiment is the causal adaptive-stiffness pipeline:

- `01_core_model/` — spring, node, geometry, and equilibrium mechanics.
- `02_baseline_profiles/` — topology and torque-profile visualizations.
- `04_adaptive_learning/` — causal MLP training and energy accounting.
  It contains both the active stiffness trainer and active preload trainer.
- `05_trajectory_evaluation/` — retained trajectory comparison utilities.
- `topologies/adaptive_stiffness/` — active baseline and 20-spring fan topologies.
- `models/adaptive_stiffness/` — current adaptive-stiffness checkpoints.
- `tables/adaptive_stiffness/` — compact train, test, and mechanics metrics.
- `plots/adaptive_stiffness/` — current figures.

The default trainer does not expose the complete torque-angle profile to the
MLP and does not write the very large per-timestep torque trace.

## Current fixed-motion experiment

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

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
  --optimizer adam `
  --learning-rate 0.01 `
  --hidden-dim 256 `
  --min-stiffness 1 `
  --max-stiffness 800 `
  --energy-weight 0.35 `
  --stiffness-change-weight 0 `
  --device cuda `
  --mechanics-backend torch `
  --mechanics-batch-size 8192 `
  --relaxation-steps 80 `
  --output-name adaptive_20spring_fixed_motion
```

Add `--write-torque-trace` only when the full per-timestep CSV is genuinely
needed; a large run can produce a trace tens or hundreds of megabytes in size.

## Organized research artifacts

Models, plots, tables, and topologies are grouped consistently:

- `adaptive_stiffness/` — current causal stiffness-control work.
- `preload/` — preserved preload studies.
- `isaaclab/` — preserved IsaacLab rollouts and exported policies.
- `legacy/` — earlier models, comparisons, and exploratory results.

`data/isaaclab/` contains the archived rollout datasets. Historical source is
under `archive/`; it is retained for reproducibility but is not part of the
active workflow.
