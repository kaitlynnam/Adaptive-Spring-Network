from simulation.block_diagrams import (
    adaptive_pej_loop_diagram,
    cam_spring_network_diagram,
    repo_workflow_diagram,
    write_default_diagrams,
)


def test_adaptive_pej_loop_diagram_renders_svg():
    svg = adaptive_pej_loop_diagram().to_svg()

    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "Adaptive PEJ Simulation Loop" in svg
    assert "Joint dynamics" in svg
    assert "solve_ivp" in svg


def test_write_default_diagrams_creates_svg_files(tmp_path):
    paths = write_default_diagrams(tmp_path)

    assert len(paths) == 5
    for path in paths:
        assert path.suffix == ".svg"
        assert path.exists()


def test_cam_spring_network_diagram_renders_requested_chain():
    svg = cam_spring_network_diagram().to_svg()

    assert "Cam-Controlled 3-Spring Adaptive PEJ Network" in svg
    assert "Cam Actuator Dynamics" in svg
    assert "Net Energy Savings" in svg


def test_repo_workflow_diagram_renders_top_level_workflow():
    svg = repo_workflow_diagram().to_svg()

    assert "Adaptive PEJ Repo Workflow" in svg
    assert "Trajectory Data" in svg
    assert "Artifacts" in svg
