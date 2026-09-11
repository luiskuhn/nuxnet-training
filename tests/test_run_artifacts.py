import json

import numpy as np
import pytest
import tifffile
import torch
import yaml

from nidavellir_tools.run_artifacts import prepare_model_package_artifacts


def _specification(root):
    (root / "network.py").write_text("# packaged architecture\n")
    (root / "environment.yml").write_text("dependencies:\n- pytorch\n")
    cover = root / "docs" / "images" / "graph_abstract_nuxnet_training.png"
    cover.parent.mkdir(parents=True)
    cover.write_bytes(b"local cover")
    path = root / "source.yaml"
    path.write_text(yaml.safe_dump({
        "type": "model",
        "covers": [{"source": "docs/images/graph_abstract_nuxnet_training.png"}],
        "weights": {"pytorch_state_dict": {
            "source": "weights.pt",
            "architecture": {"source": "network.py", "callable": "Network"},
            "dependencies": {"source": "environment.yml"},
        }}
    }))
    return path


def test_artifacts_capture_direct_model_boundary(tmp_path):
    model = torch.nn.Conv3d(1, 2, kernel_size=1, bias=False)
    sample = torch.arange(24, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    output = prepare_model_package_artifacts(
        model, sample, {"parent_model": "run-1"}, _specification(tmp_path),
        tmp_path / "stage", provenance={"id": "parent"},
    )

    np.testing.assert_array_equal(np.load(output / "test-input.npy"), sample.numpy())
    np.testing.assert_array_equal(
        np.load(output / "test-output.npy"), model(sample).detach().numpy()
    )
    sample_input = tifffile.imread(output / "sample-input.tif")
    sample_output = tifffile.imread(output / "sample-output.tif")
    assert sample_input.shape == (2, 3, 4)
    assert sample_output.shape == (2, 2, 3, 4)
    assert sample_input.dtype == sample_output.dtype == np.float32
    np.testing.assert_array_equal(sample_input, sample.numpy()[0, 0])
    np.testing.assert_array_equal(sample_output, model(sample).detach().numpy()[0])
    assert json.loads((output / "cli-parameters.json").read_text())["parent_model"] == "run-1"
    assert (
        output / "docs" / "images" / "graph_abstract_nuxnet_training.png"
    ).read_bytes() == b"local cover"
    assert {"model-package.yaml", "README.md", "weights.pt", "network.py",
            "sample-input.tif", "sample-output.tif",
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
