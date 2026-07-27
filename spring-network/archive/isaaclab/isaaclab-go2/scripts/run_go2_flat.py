"""Spawn Unitree Go2 on a flat plane in Isaac Lab.

Run with:
    isaaclab.bat -p isaaclab-go2/scripts/run_go2_flat.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_DIR / "isaaclab-go2" / "source" / "asn_go2"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


def _import_app_launcher():
    """Import Isaac Lab's app launcher across current and legacy namespaces."""

    try:
        from isaaclab.app import AppLauncher

        namespace = "isaaclab"
    except ImportError:
        from omni.isaac.lab.app import AppLauncher

        namespace = "omni.isaac.lab"

    return AppLauncher, namespace


def _import_isaaclab_modules(namespace: str):
    """Import Isaac Lab modules after the simulation app has launched."""

    if namespace == "isaaclab":
        from isaaclab.assets import Articulation
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils import configclass
    else:
        from omni.isaac.lab.assets import Articulation
        from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
        from omni.isaac.lab.sim import SimulationCfg, SimulationContext
        from omni.isaac.lab.terrains import TerrainImporterCfg
        from omni.isaac.lab.utils import configclass

    return {
        "Articulation": Articulation,
        "InteractiveScene": InteractiveScene,
        "InteractiveSceneCfg": InteractiveSceneCfg,
        "SimulationCfg": SimulationCfg,
        "SimulationContext": SimulationContext,
        "TerrainImporterCfg": TerrainImporterCfg,
        "configclass": configclass,
        "namespace": namespace,
    }


AppLauncher, ISAACLAB_NAMESPACE = _import_app_launcher()

parser = argparse.ArgumentParser(description="Spawn the Unitree Go2 on flat ground.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

isaac = _import_isaaclab_modules(ISAACLAB_NAMESPACE)
Articulation = isaac["Articulation"]
InteractiveScene = isaac["InteractiveScene"]
InteractiveSceneCfg = isaac["InteractiveSceneCfg"]
SimulationCfg = isaac["SimulationCfg"]
SimulationContext = isaac["SimulationContext"]
TerrainImporterCfg = isaac["TerrainImporterCfg"]
configclass = isaac["configclass"]

from asn_go2.robots import get_go2_cfg


@configclass
class Go2FlatSceneCfg(InteractiveSceneCfg):
    """Scene with a ground plane and one Go2 robot per environment."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        debug_vis=False,
    )

    robot = get_go2_cfg()


def main() -> None:
    """Launch the simulator and keep Go2 alive on a flat plane."""

    sim = SimulationContext(SimulationCfg(dt=0.005))
    sim.set_camera_view(eye=(3.5, 3.5, 2.5), target=(0.0, 0.0, 0.4))

    scene = InteractiveScene(Go2FlatSceneCfg(num_envs=1, env_spacing=2.0))
    robot: Articulation = scene["robot"]

    sim.reset()
    print(f"[INFO] Using Isaac Lab namespace: {isaac['namespace']}")
    print("[INFO] Spawned Unitree Go2 on flat ground.")

    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
