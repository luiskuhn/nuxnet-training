"""Capture the reproducible, run-specific inputs to model packaging."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nidavellir_tools.sample_tensors import create_sample_tensor

__all__ = ["prepare_model_package_artifacts"]


def _json_value(value: Any) -> Any:
    """Convert common argparse values without silently stringifying arbitrary objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"CLI parameter is not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _copy_rdf_dependencies(rdf: dict[str, Any], base: Path, output: Path) -> None:
    state = rdf.get("weights", {}).get("pytorch_state_dict", {})
    descriptors = (
        state.get("architecture"),
        state.get("dependencies"),
        *rdf.get("covers", []),
    )
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("source"), str):
            continue
        relative = Path(descriptor["source"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"RDF artifact source must be a safe relative path: {relative}")
        source = base / relative
        if not source.is_file():
            raise FileNotFoundError(f"RDF artifact not found: {source}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def prepare_model_package_artifacts(
    model,
    sample_input,
    cli_parameters,
    specification_path,
    output_dir,
    model_card_path=None,
    provenance=None,
    sample_layout="auto",
    sample_batch_index=0,
    sample_channel_policy="squeeze-singleton",
) -> Path:
    """Stage a fitted bare PyTorch model and its exact direct inference pair."""
    specification_path, output_dir = Path(specification_path), Path(output_dir)
    parameters = _json_value(dict(cli_parameters))
    overwrite = bool(
        parameters.get("overwrite_model_package", parameters.get("overwrite", False))
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rdf = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
    state = rdf.get("weights", {}).get("pytorch_state_dict") if isinstance(rdf, dict) else None
    if not isinstance(state, dict):
        raise ValueError("specification requires weights.pytorch_state_dict")
    if set(rdf.get("weights", {})) != {"pytorch_state_dict"}:
        raise ValueError("prototype supports only pytorch_state_dict weights")

    shutil.copy2(specification_path, output_dir / "model-package.yaml")
    _copy_rdf_dependencies(rdf, specification_path.parent, output_dir)
    if model_card_path is None:
        shutil.copy2(
            Path(__file__).with_name("model-card-template.md"), output_dir / "README.md"
        )
    else:
        shutil.copy2(Path(model_card_path), output_dir / "README.md")

    weights = output_dir / "weights.pt"
    torch.save(model.state_dict(), weights)  # nosec B614: tensor state only
    input_array = sample_input.detach().cpu().numpy()
    np.save(output_dir / "test-input.npy", input_array)
    device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    with torch.inference_mode():
        raw_output = model(sample_input.to(device))
    output_array = raw_output.detach().cpu().numpy()
    np.save(output_dir / "test-output.npy", output_array)

    # The NPY files above are the exact model boundary and remain authoritative.
    # TIFFs below are separate, RDF-driven presentation views of one batch item.
    inputs, outputs = rdf.get("inputs"), rdf.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("artifact staging requires exactly one RDF input tensor")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("artifact staging requires exactly one RDF output tensor")
    sample_records = []
    for array, tensor, fallback in (
        (input_array, inputs[0], "sample-input.tif"),
        (output_array, outputs[0], "sample-output.tif"),
    ):
        descriptor = tensor.get("sample_tensor", {})
        filename = descriptor.get("source", fallback)
        sample_records.append(
            create_sample_tensor(
                array,
                tensor.get("axes"),
                output_dir / filename,
                layout=sample_layout,
                batch_index=sample_batch_index,
                channel_policy=sample_channel_policy,
            )
        )
    (output_dir / "cli-parameters.json").write_text(
        json.dumps(parameters, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )

    record = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": _git_commit(specification_path.parent),
        "weights_sha256": _sha256(weights),
        "input": {"shape": list(input_array.shape), "dtype": str(input_array.dtype)},
        "output": {"shape": list(output_array.shape), "dtype": str(output_array.dtype)},
        "sample_tensors": sample_records,
    }
    if provenance:
        parent = (
            json.loads(Path(provenance).read_text(encoding="utf-8"))
            if isinstance(provenance, (str, Path))
            else provenance
        )
        record["parent_model"] = _json_value(parent)
    (output_dir / "run-provenance.json").write_text(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir
