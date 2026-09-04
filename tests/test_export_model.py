import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml


SCRIPT = Path(__file__).parents[1] / "nidavellir_tools" / "build_model_package.py"
SPEC = importlib.util.spec_from_file_location("build_model_package", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_packaging_reference_is_valid_yaml_and_has_required_contract_sections():
    reference = yaml.safe_load(
        (SCRIPT.parent / "examples" / "model-package.example.yaml").read_text()
    )

    assert reference["type"] == "model"
    assert reference["documentation"]["source"] == "README.md"
    assert reference["inputs"][0]["test_tensor"]["source"]
    assert reference["outputs"][0]["test_tensor"]["source"]
    assert reference["weights"]["pytorch_state_dict"]["architecture"]["callable"]


def test_builds_domain_independent_repository_package(tmp_path):
    architecture = tmp_path / "network.py"
    architecture.write_text(
        "import torch\nclass Network(torch.nn.Linear):\n"
        "    def __init__(self, features=2):\n"
        "        super().__init__(features, features, bias=False)\n",
        encoding="utf-8",
    )
    environment = tmp_path / "environment.yml"
    environment.write_text("dependencies:\n- pytorch\n", encoding="utf-8")
    specification = tmp_path / "model.yaml"
    specification.write_text(yaml.safe_dump({
        "format_version": "0.5.3", "type": "model", "name": "Generic vision model",
        "version": "1.0.0", "description": "No domain assumptions.",
        "documentation": {"source": "README.md"},
        "inputs": [{"id": "input", "test_tensor": {"source": "test-input.npy"}}],
        "outputs": [{"id": "output", "test_tensor": {"source": "test-output.npy"}}],
        "weights": {
            "pytorch_state_dict": {
                "source": "weights.pt",
                "architecture": {"source": "network.py", "callable": "Network",
                                 "kwargs": {"features": 2}},
                "dependencies": {"source": "environment.yml"},
            },
            "torchscript": {"source": "model.ts", "parent": "pytorch_state_dict"},
        },
    }, sort_keys=False), encoding="utf-8")
    checkpoint = tmp_path / "trained.ckpt"
    network = torch.nn.Linear(2, 2, bias=False)
    torch.save({"state_dict": {
        f"network.{key}": value for key, value in network.state_dict().items()
    }}, checkpoint)
    test_input = tmp_path / "input.npy"
    test_output = tmp_path / "output.npy"
    np.save(test_input, np.zeros((1, 2), dtype=np.float32))
    np.save(test_output, np.zeros((1, 2), dtype=np.float32))
    card = tmp_path / "card.md"
    card.write_text("# Generic model\n", encoding="utf-8")
    provenance = tmp_path / "run.json"
    provenance.write_text(json.dumps({"training": {"run_id": "run-123"}}), encoding="utf-8")

    package, archive = builder.build_model_package(
        specification, checkpoint, test_input, test_output, card, tmp_path / "package",
        provenance=provenance, state_dict_key="state_dict", strip_prefix="network.",
        trace_input=test_input,
    )
    rdf = yaml.safe_load((package / "rdf.yaml").read_text())
    recorded = json.loads((package / "provenance.json").read_text())

    assert archive.is_file()
    assert {"rdf.yaml", "README.md", "weights.pt", "model.ts", "test-input.npy",
            "test-output.npy", "network.py", "environment.yml", "provenance.json",
            "SHA256SUMS"} <= {path.name for path in package.iterdir()}
    assert rdf["weights"]["pytorch_state_dict"]["sha256"] == builder._digest(package / "weights.pt")
    assert rdf["weights"]["torchscript"]["sha256"] == builder._digest(package / "model.ts")
    assert recorded["training"]["run_id"] == "run-123"
    assert recorded["training"]["checkpoint_sha256"] == builder._digest(checkpoint)


def test_rejects_nonempty_output_without_overwrite(tmp_path):
    import pytest

    output = tmp_path / "package"
    output.mkdir()
    (output / "keep.txt").write_text("do not delete")
    with pytest.raises(FileExistsError, match="output directory is not empty"):
        builder.build_model_package(
            tmp_path / "missing.yaml", tmp_path / "missing.pt", tmp_path / "input.npy",
            tmp_path / "output.npy", tmp_path / "README.md", output,
        )
