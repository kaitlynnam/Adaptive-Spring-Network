from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    name: str
    type: str
    local_position: np.ndarray
    current_position: np.ndarray | None = None


def as_vector(position):
    return np.asarray(position, dtype=float)


def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate(position, theta):
    return rotation_matrix(theta) @ as_vector(position)


def update_node_position(node, theta):
    """Update a node from its local frame into the current world frame.

    Limb 2 nodes rotate with the joint angle. Limb 1 nodes, fixed anchors, and
    simple internal nodes are held in world coordinates in this first model.
    Later, this is where additional kinematics or quasi-static node relaxation
    could be added.
    """
    if node.type == "limb2":
        node.current_position = rotate(node.local_position, theta)
    else:
        node.current_position = as_vector(node.local_position)

    return node.current_position
