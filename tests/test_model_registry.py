import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


SCRIPT = Path(__file__).parents[1] / "nidavellir_tools" / "model_package_registry.py"
SPEC = importlib.util.spec_from_file_location("model_registry", SCRIPT)
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)


def make_package(path: Path) -> Path:
    path.mkdir()
    model = torch.jit.trace(torch.nn.Identity(), torch.zeros(1, 2))
    model.save(str(path / "model.ts"))
    rdf = {
        "format_version": "0.5.3",
        "type": "model",
        "name": "framework-neutral test",
        "weights": {"torchscript": {
            "source": "model.ts", "sha256": registry._digest(path / "model.ts")
        }},
    }
    (path / "rdf.yaml").write_text(yaml.safe_dump(rdf), encoding="utf-8")
    return path


def test_stage_verify_and_load_local_package(tmp_path):
    source = make_package(tmp_path / "source")
    staged = registry.stage(str(source), tmp_path / "staged")

    rdf = registry.verify(staged)
    model = registry.load(staged)

    assert rdf["name"] == "framework-neutral test"
    assert torch.equal(model(torch.tensor([[1.0, 2.0]])), torch.tensor([[1.0, 2.0]]))


def test_stage_zip_and_reject_checksum_tampering(tmp_path):
    source = make_package(tmp_path / "source")
    archive = Path(__import__("shutil").make_archive(str(tmp_path / "model"), "zip", source))
    staged = registry.stage(str(archive), tmp_path / "staged")
    (staged / "model.ts").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        registry.verify(staged)


def test_safe_extract_rejects_path_traversal(tmp_path):
    import zipfile

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        registry.stage(str(archive), tmp_path / "staged")


def test_loads_rdf_state_dict_with_packaged_architecture(tmp_path):
    package = tmp_path / "state-package"
    package.mkdir()
    architecture = package / "architecture.py"
    architecture.write_text(
        "import torch\nclass Network(torch.nn.Linear):\n"
        "    def __init__(self, features=2):\n"
        "        super().__init__(features, features, bias=False)\n",
        encoding="utf-8",
    )
    weights = package / "weights.pt"
    torch.save(torch.nn.Linear(2, 2, bias=False).state_dict(), weights)
    rdf = {
        "type": "model",
        "weights": {"pytorch_state_dict": {
            "source": "weights.pt", "sha256": registry._digest(weights),
            "architecture": {
                "source": "architecture.py", "sha256": registry._digest(architecture),
                "callable": "Network", "kwargs": {"features": 2},
            },
        }},
    }
    (package / "rdf.yaml").write_text(yaml.safe_dump(rdf), encoding="utf-8")

    model = registry.load(package, representation="pytorch_state_dict")

    assert isinstance(model, torch.nn.Linear)
    assert model.training is False


def test_exported_child_preserves_specification_and_reloads_as_parent(tmp_path):
    package = tmp_path / "parent"
    package.mkdir()
    architecture = package / "architecture.py"
    architecture.write_text(
        "import torch\nclass Network(torch.nn.Linear):\n"
        "    def __init__(self, features=2):\n"
        "        super().__init__(features, features, bias=False)\n",
        encoding="utf-8",
    )
    parent_model = torch.nn.Linear(2, 2, bias=False)
    weights = package / "weights.pt"
    torch.save(parent_model.state_dict(), weights)
    test_input = package / "test-input.npy"
    test_output = package / "test-output.npy"
    __import__("numpy").save(test_input, __import__("numpy").zeros((1, 2)))
    __import__("numpy").save(test_output, __import__("numpy").zeros((1, 2)))
    card = package / "README.md"
    card.write_text("# Parent\n", encoding="utf-8")
    rdf = {
        "format_version": "0.5.3", "type": "model", "name": "Parent", "version": "1",
        "description": "parent model", "documentation": {
            "source": "README.md", "sha256": registry._digest(card),
        },
        "inputs": [{"test_tensor": {
            "source": "test-input.npy", "sha256": registry._digest(test_input),
        }}],
        "outputs": [{"test_tensor": {
            "source": "test-output.npy", "sha256": registry._digest(test_output),
        }}],
        "weights": {"pytorch_state_dict": {
            "source": "weights.pt", "sha256": registry._digest(weights),
            "architecture": {
                "source": "architecture.py", "sha256": registry._digest(architecture),
                "callable": "Network", "kwargs": {"features": 2},
            },
        }},
    }
    (package / "rdf.yaml").write_text(yaml.safe_dump(rdf), encoding="utf-8")
    (package / "provenance.json").write_text(
        json.dumps({"model": {"version": "1"}}), encoding="utf-8"
    )
    child_checkpoint = tmp_path / "trained.ckpt"
    child_model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        child_model.weight.fill_(3.0)
    torch.save({"state_dict": {
        f"model.{key}": value for key, value in child_model.state_dict().items()
    }}, child_checkpoint)
    child_output = tmp_path / "child-output.npy"
    __import__("numpy").save(child_output, __import__("numpy").full((1, 2), 6.0))
    child_card = tmp_path / "README-child.md"
    child_card.write_text("# Child\n\nTrained on the new dataset.\n", encoding="utf-8")

    child = registry.export_child_package(
        package, child_checkpoint, child_output, child_card, tmp_path / "child",
        version="2", parent_identifier="hf://owner/parent@commit",
        state_dict_key="state_dict", strip_prefix="model.",
    )
    loaded = registry.load_package(child, representation="pytorch_state_dict")

    assert loaded.rdf["version"] == "2"
    assert loaded.provenance["parent"]["identifier"] == "hf://owner/parent@commit"
    assert loaded.provenance["parent"]["version"] == "1"
    assert loaded.provenance["training"]["checkpoint_sha256"] == registry._digest(child_checkpoint)
    assert torch.equal(loaded.model.weight, child_model.weight)
    assert (child / "README.md").read_text() == child_card.read_text()
    assert child.with_suffix(".zip").is_file()
