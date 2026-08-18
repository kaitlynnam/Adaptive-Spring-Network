# Causal Period-Buffer Spring Controller

The active controller updates stiffness once per motion period. Period 1 uses
the topology defaults. At each later boundary, an MLP consumes the preceding
period's 160 samples of six measured channels and predicts one stiffness for
each of the 60 springs. The vector is fixed throughout the next period.

## Active layout

- `01_core_model/mechanics_3d.py` — relaxed 3D equilibrium and joint torque.
- `01_core_model/audit_spatial_feasibility.py` — topology clearance audit.
- `04_adaptive_learning/train_period_adaptive_3d.py` — closed-loop trainer.
- `04_adaptive_learning/deploy_period_adaptive_3d.py` — causal deployment.
- `04_adaptive_learning/benchmark_period_adaptive_deployment.py` — many-profile benchmark.
- `04_adaptive_learning/generate_period_adaptive_simulation.py` — interactive viewer.
- `topologies/spatial/` — active split-skin 60-spring topology.
- `models/period_adaptive_3d/` — period-controller checkpoints.
- `plots/period_adaptive_3d/` and `tables/period_adaptive_3d/` — active results.

The single active reference model and output prefix is
`period_adaptive_3d_60spring_bounded_extended`. Earlier period-controller
ablations and tuning runs are archived.

## Causal sequence

```text
default stiffness -> run Period 1 -> buffer motion and torque
                  -> predict stiffness -> hold during Period 2
                  -> buffer Period 2 -> predict stiffness for Period 3 -> ...
```

The network input has 960 values: 160 samples each of angle, velocity,
acceleration, target torque, spring torque, and residual motor torque.

## Usage

Run commands from the repository root:

```powershell
python spring-network/04_adaptive_learning/train_period_adaptive_3d.py
python spring-network/04_adaptive_learning/deploy_period_adaptive_3d.py
python spring-network/04_adaptive_learning/benchmark_period_adaptive_deployment.py
python spring-network/04_adaptive_learning/generate_period_adaptive_simulation.py
python -m pytest spring-network/tests -q
```

The environment is pinned in `../environment.yml`. Historical pipelines and
their artifacts are under `archive/`; they are not imported by active code.
