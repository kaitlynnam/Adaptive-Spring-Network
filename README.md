# Adaptive Spring Network

This repository contains one active research pipeline: a causal 3D controller
that buffers one complete motion period, predicts 60 spring stiffnesses at the
period boundary, and holds those stiffnesses throughout the following period.

```text
completed period -> 960-value observation -> MLP -> 60 stiffnesses -> next period
```

The first period always uses the topology's default stiffness vector. The
controller observes angle, angular velocity, angular acceleration, target
torque, realized spring torque, and residual motor torque.

## Commands

Train:

```powershell
python spring-network/04_adaptive_learning/train_period_adaptive_3d.py
```

Deploy a checkpoint:

```powershell
python spring-network/04_adaptive_learning/deploy_period_adaptive_3d.py
```

Create the interactive 3D viewer:

```powershell
python spring-network/04_adaptive_learning/generate_period_adaptive_simulation.py
```

Run the active tests:

```powershell
python -m pytest spring-network/tests -q
```

See [`spring-network/README.md`](spring-network/README.md) for the pipeline and
output layout. Superseded passive, timestep-adaptive, preload, and other
research branches are retained under `spring-network/archive/`.

Deferred research and mentor-suggested comparisons are tracked in
[`TODO.md`](TODO.md).

Research project with The University of Bristol.
