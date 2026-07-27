# Simulator Trajectory Schema

Use this schema when exporting rollout data from any simulator or controller.
The PEJ tools are simulator-agnostic: they only need joint state and the active
torque demand before passive spring assistance.

## Required Fields

| Field | Unit | Meaning |
| --- | --- | --- |
| `time` | s | sample time |
| `joint_name` | string | joint identifier, for example `front_thigh` |
| `theta` | rad | joint angle |
| `theta_dot` | rad/s | joint angular velocity |
| `tau_total` | N m | active torque demand before passive spring assistance |

## Optional Fields

| Field | Meaning |
| --- | --- |
| `terrain` | terrain or rollout condition label |
| `policy` | policy/controller/checkpoint label |
| `robot_id` | robot/environment instance id |

## Torque Convention

`tau_total` should be the active joint torque that the motor/controller would
need before adding a passive spring:

```text
tau_motor_after_pej = tau_total - tau_spring
```

Do not export a residual torque that already includes spring assistance. That
would double-count the PEJ offload.

## Loading Data

CSV and NPZ files can be loaded with:

```python
import pej

trajectory = pej.load_trajectory("rollout.npz")
joint = trajectory.for_joint("front_thigh")
```

The rest of the repo can then distill fixed PEJ profiles, run adaptive models,
or evaluate cam-controlled spring networks from the same trajectory arrays.
