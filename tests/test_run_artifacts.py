import json

import numpy as np
import pytest
import torch
import yaml

from nidavellir_tools.run_artifacts import prepare_model_package_artifacts


def _specification(root):
    (root / "network.py").write_text("# packaged architecture\n")
    (root / "environment.yml").write_text("dependencies:\n- pytorch\n")
    path = root / "source.yaml"
    path.write_text(yaml.safe_dump({
        "type": "model", "weights": {"pytorch_state_dict": {
            "source": "weights.pt",
            "architecture": {"source": "network.py", "callable": "Network"},
            "dependencies": {"source": "environment.yml"},
        }}
    }))
    return path


def test_artifacts_capture_direct_model_boundary(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    sample = torch.tensor([[1.0, 4.0]], dtype=torch.float32)
    output = prepare_model_package_artifacts(
        model, sample, {"parent_model": "run-1"}, _specification(tmp_path),
        tmp_path / "stage", provenance={"id": "parent"},
    )

    np.testing.assert_array_equal(np.load(output / "test-input.npy"), sample.numpy())
    np.testing.assert_array_equal(
        np.load(output / "test-output.npy"), model(sample).detach().numpy()
    )
    assert json.loads((output / "cli-parameters.json").read_text())["parent_model"] == "run-1"
    assert {"model-package.yaml", "README.md", "weights.pt", "network.py",
            "environment.yml", "run-provenance.json"} <= {
                path.name for path in output.rglob("*")
            }


def test_artifacts_do_not_overwrite_by_default(tmp_path):
    destination = tmp_path / "stage"
    destination.mkdir()
    (destination / "existing").touch()
    with pytest.raises(FileExistsError):
        prepare_model_package_artifacts(
            torch.nn.Linear(2, 2), torch.zeros(1, 2), {},
            _specification(tmp_path), destination,
        )
