"""Unitree Go2 configuration accessors.

Isaac Lab ships a Go2 asset configuration. This module keeps that dependency in
one place so task code can import a project-local helper instead of depending on
the exact upstream module path everywhere.
"""

from __future__ import annotations


def get_go2_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot"):
    """Return Isaac Lab's Unitree Go2 articulation config with a custom prim path."""

    try:
        from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
    except ImportError:
        from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG

    return UNITREE_GO2_CFG.replace(prim_path=prim_path)
