# Adaptive Spring Network Research Code

The active project learns to update spring stiffness once per motion period.
The first period uses the 3D topology's initial stiffnesses with no neural
network input. At its end, the complete period of angle, velocity,
acceleration, target torque, spring torque, and residual motor torque is fed to
the MLP. Its 60 predicted stiffnesses are held fixed throughout the next
period. Training aligns every completed-period input with loss on the following
period, preserving strict causality.

## Active pipeline

- `01_core_model/` — 2D and genuine-3D equilibrium mechanics.
- `04_adaptive_learning/train_period_adaptive_3d.py` — active causal trainer.
- `topologies/spatial/` — active 60-spring topology.
- `models/period_adaptive_3d/` — active checkpoints.

Run from the repository root with:

```powershell
python spring-network/04_adaptive_learning/train_period_adaptive_3d.py
```

Training learns a complete-period-data-to-stiffness mapping. By default it trains this mapping in a
six-period closed loop: half the trajectories start at default stiffness, half
start randomized, and every later period uses the NN's own preceding output.
Exact mechanics refreshes can rebuild the local torque basis between phases.
The first-period default and one-period delay are enforced by the deployment
wrapper:

```powershell
python spring-network/04_adaptive_learning/deploy_period_adaptive_3d.py
```

Continue a checkpoint on a larger dataset with, for example:

```powershell
python spring-network/04_adaptive_learning/train_period_adaptive_3d.py `
  --resume-checkpoint spring-network/models/period_adaptive_3d/period_adaptive_3d_60spring.npz `
  --training-profiles 12000 --iterations 5000 --learning-rate 0.0003 `
  --output-name period_adaptive_3d_60spring_extended
```

Every completed training run performs held-out relaxed-3D evaluation and writes:

- `plots/period_adaptive_3d/*_training_convergence.png`
- `plots/period_adaptive_3d/*_torque_time.png`
- `plots/period_adaptive_3d/*_torque_angle.png`
- `plots/period_adaptive_3d/*_stiffness_schedule.png`
- `tables/period_adaptive_3d/*_summary.csv`

Create an interactive standalone HTML deployment simulation with:

```powershell
python 04_adaptive_learning/generate_period_adaptive_simulation.py `
  --checkpoint models/period_adaptive_3d/period_adaptive_3d_60spring_closed_loop_long.npz
```

Custom input CSV files require `time_s,target_torque_nm`; `angle_deg` is
optional. Pass one with `--input-csv path/to/trajectory.csv`. The viewer
animates the relaxed 3D spring network, torque traces, period cursor, and the
60-value stiffness vector as updates occur at period boundaries.

The profile-conditioned passive experiment remains available for comparison.
The former timestep-adaptive, preload, trajectory-evaluation, and paper
pipeline is preserved under `archive/timestep_adaptation/`.

## Reproducible CUDA environment

The tested Windows environment is pinned in `environment.yml`. Create it with:

```powershell
conda env create -f environment.yml
```

Run tests with `python -m pytest spring-network/tests -q`.
