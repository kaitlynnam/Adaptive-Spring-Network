# Isaac Lab Rollout Export

Use these scripts to replay a pretrained Isaac Lab/RSL-RL checkpoint and export
real joint trajectories into the active `spring-network` workflow.

## 1. Export A Checkpoint Rollout

Run from the repository root with an Isaac Lab-enabled Python:

```powershell
<ISAAC_LAB_ROOT>\isaaclab.bat -p spring-network\06_isaaclab_export\export_pretrained_rollout.py `
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 `
  --use-pretrained-checkpoint `
  --num-envs 64 `
  --steps 2000 `
  --headless
```

Outputs are written to:

```text
spring-network/data/isaaclab_rollouts/
```

The exporter writes:

- `.npz`: compact batched rollout arrays for all envs and joints.
- `.csv`: long-form rows with one row per step/env/joint.
- `_metadata.json`: task, checkpoint, joint names, torque source, and timing.

New exports also preserve Isaac Lab's native per-environment terrain metadata:
`terrain_family`, `terrain_type_index`, `terrain_level`, `terrain_origin`, and
the auditable column-to-family mapping `terrain_column_families`.

To train the internal-fan knee model on only one terrain family:

```powershell
python spring-network\06_isaaclab_export\train_internal_fan_multi_rollout.py `
  spring-network\data\isaaclab_rollouts\<rollout>.npz `
  --joint FL_calf_joint `
  --terrain-family random_rough `
  --output-name internal_fan_go2_fl_knee_random_rough `
  --device cuda
```

By default, `tau_total` uses Isaac Lab's applied actuator torque. To export the
pre-saturation actuator demand instead, run with:

```powershell
--torque-source computed
```

On this machine, the rough Go2 pretrained RSL-RL checkpoint is also available
directly at:

```text
C:\Users\kn109\Code\IsaacLab\.pretrained_checkpoints\rsl_rl\Isaac-Velocity-Rough-Unitree-Go2-v0\checkpoint.pt
```

## 2. Extract One Joint For Spring-Network Evaluation

The current spring-network evaluator expects one joint trajectory with
`t, theta, theta_dot, tau_target`. Extract one joint from the rollout NPZ:

```powershell
python spring-network\06_isaaclab_export\extract_single_joint_csv.py `
  spring-network\data\isaaclab_rollouts\Isaac-Velocity-Flat-Unitree-Go2-v0_model_1000.npz `
  --joint FL_thigh_joint `
  --env-id 0
```

The single-joint CSV is written to:

```text
spring-network/data/isaaclab_rollouts/single_joint/
```

## 3. Evaluate The Spring Network On That Trajectory

```powershell
python spring-network\05_trajectory_evaluation\evaluate_trajectory.py `
  --trajectory spring-network\data\isaaclab_rollouts\single_joint\<exported_joint>.csv `
  --single
```

Use the joint names from the exporter's `_metadata.json` if your Go2 task uses
different names.
