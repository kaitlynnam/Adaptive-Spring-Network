import json
from pathlib import Path

from geometry import Node, as_vector
from network import SpringNetwork
from physics import Spring


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY_PATH = PROJECT_ROOT / "topologies" / "profile_conditioned_passive" / "baseline_model.json"


def _load_topology_data(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "base_topology" not in data:
        return data
    base = _load_topology_data(path.parent / data["base_topology"])
    node_overrides = data.get("node_overrides", {})
    for node in base["nodes"]:
        if node["name"] in node_overrides:
            node.update(node_overrides[node["name"]])
    stiffness_scale = float(data.get("stiffness_scale", 1.0))
    for spring in base["springs"]:
        spring["stiffness_k"] *= stiffness_scale
    base["rest_length_scale"] = float(base.get("rest_length_scale", 1.0)) * float(
        data.get("rest_length_scale", 1.0)
    )
    base["name"] = data.get("name", base["name"])
    base["description"] = data.get("description", base.get("description", ""))
    base["derived_from"] = data["base_topology"]
    return base


def load_network(path=DEFAULT_TOPOLOGY_PATH):
    """Load a spring network topology from a JSON file.

    The file defines the network structure: node names/types/positions and the
    spring pairs connecting them. This keeps hand-authored or future generated
    topologies separate from the physics code.
    """
    path = Path(path)
    data = _load_topology_data(path)

    return build_network_from_topology(data), data


def build_network_from_topology(data):
    """Build a SpringNetwork from an already-loaded topology dictionary."""
    nodes = {
        item["name"]: Node(
            name=item["name"],
            type=item["type"],
            local_position=as_vector(item["position"]),
        )
        for item in data["nodes"]
    }

    springs = [
        Spring(
            node_a=item["node_a"],
            node_b=item["node_b"],
            stiffness_k=float(item["stiffness_k"]),
            rest_length=item.get("rest_length"),
        )
        for item in data["springs"]
    ]

    validate_topology(nodes, springs)
    network = SpringNetwork(nodes=nodes, springs=springs)

    if any(spring.rest_length is None for spring in springs):
        network.initialize_rest_lengths(theta=float(data.get("rest_angle_degrees", 0.0)) * 3.141592653589793 / 180.0)
    rest_length_scale = float(data.get("rest_length_scale", 1.0))
    if rest_length_scale != 1.0:
        for spring in springs:
            spring.rest_length *= rest_length_scale

    return network


def save_topology(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def validate_topology(nodes, springs):
    valid_types = {"fixed", "limb1", "limb2", "internal"}
    for node in nodes.values():
        if node.type not in valid_types:
            raise ValueError(f"Node {node.name} has unsupported type {node.type!r}.")

    for spring in springs:
        if spring.node_a not in nodes:
            raise ValueError(f"Spring references missing node {spring.node_a!r}.")
        if spring.node_b not in nodes:
            raise ValueError(f"Spring references missing node {spring.node_b!r}.")
        if spring.node_a == spring.node_b:
            raise ValueError(f"Spring cannot connect node {spring.node_a!r} to itself.")
        if spring.stiffness_k <= 0.0:
            raise ValueError(f"Spring {spring.node_a}-{spring.node_b} must have positive stiffness.")
