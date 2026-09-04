from pathlib import Path

import yaml


def test_mlproject_exposes_and_forwards_anisotropic_training_parameters():
    project = yaml.safe_load((Path(__file__).parents[1] / "MLproject").read_text())
    main = project["entry_points"]["main"]
    parameters = main["parameters"]
    expected_defaults = {
        "patch-size": "32,128,128",
        "target-voxel-spacing": "3.0,1.0,1.0",
        "random-rotation-degrees": 10.0,
        "random-rotation-90-probability": 0.5,
        "ce-loss-weight": 1.0,
        "dice-loss-weight": 1.0,
        "class-weights": "1.0,1.0",
        "initial-weights": "none",
        "parent-metadata": "none",
        "inference-overlap": 0.5,
    }

    for name, default in expected_defaults.items():
        assert parameters[name]["default"] == default
        assert f"--{name} {{{name}}}" in main["command"]

    assert "loss-function" not in parameters
    assert "--loss-function" not in main["command"]
