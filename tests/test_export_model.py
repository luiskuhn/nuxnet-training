import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml

from numorph_nuclei_segmentation.model import UNet3D


SCRIPT = Path(__file__).parents[1] / "tools" / "export_model.py"
SPEC = importlib.util.spec_from_file_location("export_model", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_exports_repository_ready_fair_package(tmp_path):
    checkpoint = tmp_path / "trained.pt"
    torch.save(UNet3D(dropout=0.1).state_dict(), checkpoint)
    args = argparse.Namespace(
        checkpoint=str(checkpoint), output_dir=str(tmp_path / "package"), name="Test NuxNet",
        description="A test nucleus-marker model.", author="Test Author", author_orcid="0000-0000-0000-0000",
        github_user="test-author", license="MIT", citation="Test et al.", doi="10.0000/example",
        citation_url=None, dataset="example/dataset", dataset_version="1.0", model_version="1.2.3",
        mlflow_run_id="run-123", source_repository="https://example.org/repo", git_commit="abc123", cover=None,
        input_channels=1, classes=2, dropout=0.1, normalize_input=True, test_shape=(4, 8, 8), overwrite=False,
    )

    package, archive = exporter.export_model(args)
    rdf = yaml.safe_load((package / "rdf.yaml").read_text())
    provenance = json.loads((package / "provenance.json").read_text())

    assert archive.is_file()
    assert {"rdf.yaml", "README.md", "ARTIFACTS.md", "cover.png", "weights.pt", "model.ts", "test-input.npy", "test-output.npy", "SHA256SUMS"} <= {p.name for p in package.iterdir()}
    assert rdf["format_version"] == "0.5.14"
    assert rdf["inputs"][0]["preprocessing"][1]["id"] == "scale_range"
    assert rdf["outputs"][0]["postprocessing"][0]["id"] == "softmax"
    assert rdf["covers"][0]["sha256"] == exporter.sha256(package / "cover.png")
    assert rdf["weights"]["pytorch_state_dict"]["sha256"] == exporter.sha256(package / "weights.pt")
    assert provenance["training"]["mlflow_run_id"] == "run-123"
    assert "pipeline_tag: image-segmentation" in (package / "README.md").read_text()
    assert "`rdf.yaml`" in (package / "ARTIFACTS.md").read_text()


def test_rejects_nonempty_output_without_overwrite(tmp_path):
    import pytest

    output = tmp_path / "package"
    output.mkdir()
    (output / "keep.txt").write_text("do not delete")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not read because the output guard runs first")
    args = argparse.Namespace(checkpoint=str(checkpoint), output_dir=str(output), overwrite=False)
    with pytest.raises(FileExistsError, match="output directory is not empty"):
        exporter.export_model(args)
