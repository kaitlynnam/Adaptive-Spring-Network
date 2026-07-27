# Isaac Lab Unitree Go2

This directory is a small Isaac Lab workspace for launching the Unitree Go2 robot
from Isaac Lab's built-in assets.

It does not vendor Isaac Sim, Isaac Lab, or robot USD assets. Install Isaac Lab
separately, then run these scripts from an Isaac Lab-enabled Python environment.

## Layout

```text
isaaclab-go2/
  extension.toml                 Isaac Lab extension metadata
  pyproject.toml                  Editable package metadata
  source/asn_go2/asn_go2/         Python package for local Go2 simulation code
  scripts/run_go2_flat.py         Smoke test: spawn Go2 on a flat plane
```

## Quick Start

From your Isaac Lab checkout or an environment where `isaaclab` is importable:

```powershell
cd C:\Users\kn109\Code\Adaptive-Spring-Network
<ISAAC_LAB_ROOT>\isaaclab.bat -p isaaclab-go2\scripts\run_go2_flat.py
```

To run headless:

```powershell
<ISAAC_LAB_ROOT>\isaaclab.bat -p isaaclab-go2\scripts\run_go2_flat.py --headless
```

If your Isaac Lab install still uses the older `omni.isaac.lab` namespace, the
script includes import fallbacks for that API line.

## Next Steps

1. Confirm the smoke test launches and the Go2 stands on the plane.
2. Add terrain, commands, observations, rewards, and reset logic under
   `source/asn_go2/asn_go2/tasks/`.
3. Connect this repo's adaptive spring-network outputs to an Isaac Lab
   controller or actuator model.
