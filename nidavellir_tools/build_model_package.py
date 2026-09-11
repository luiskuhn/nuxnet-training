#!/usr/bin/env python3
"""Build a FAIR BioImage.IO/Hugging Face package from a declarative RDF template."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nidavellir_tools.model_package_registry import (  # noqa: E402
    _checkpoint_state,
    _digest,
    _resolve_callable,
    _write_zip,
    verify,
)


def _descriptors(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("source"), str):
            yield value
        for nested in value.values():
            yield from _descriptors(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _descriptors(nested)


def _copy_declared_artifacts(rdf: dict[str, Any], base: Path, output: Path) -> None:
    """Copy local template artifacts, preserving their RDF-relative locations."""
    state_descriptor = rdf.get("weights", {}).get("pytorch_state_dict", {})
    descriptors = [
        state_descriptor.get("architecture"),
        state_descriptor.get("dependencies"),
        *rdf.get("covers", []),
    ]
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("source"), str):
            continue
        relative = Path(descriptor["source"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"RDF artifact source must be a safe relative path: {relative}")
        source = base / relative
        destination = output / relative
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    sample_descriptors = (
        tensor.get("sample_tensor")
        for field in ("inputs", "outputs")
        for tensor in rdf.get(field, [])
        if isinstance(tensor, dict)
    )
    for descriptor in sample_descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("source"), str):
            raise ValueError("every RDF tensor requires a local sample_tensor source")
        relative = Path(descriptor["source"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"RDF artifact source must be a safe relative path: {relative}")
        source = base / relative
        if not source.is_file():
            raise FileNotFoundError(f"declared sample_tensor artifact not found: {source}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _refresh_hashes(rdf: dict[str, Any], output: Path) -> None:
    for descriptor in _descriptors(rdf):
        artifact = output / descriptor["source"]
        if artifact.is_file():
            descriptor["sha256"] = _digest(artifact)


def _tensor_descriptors(rdf: dict[str, Any], field: str) -> list[dict[str, Any]]:
    tensors = rdf.get(field)
    if not isinstance(tensors, list) or not tensors:
        raise ValueError(f"specification requires at least one RDF {field} tensor")
    descriptors = [tensor.get("test_tensor") for tensor in tensors]
    if not all(
        isinstance(descriptor, dict) and isinstance(descriptor.get("source"), str)
        for descriptor in descriptors
    ):
        raise ValueError(f"every RDF {field} tensor requires a local test_tensor source")
    return descriptors


def _as_paths(value: Path | list[Path]) -> list[Path]:
    return value if isinstance(value, list) else [value]


def build_model_package(
    specification: Path,
    checkpoint: Path,
    test_input: Path | list[Path],
    test_output: Path | list[Path],
    model_card: Path,
    output: Path,
    *,
    provenance: Path | None = None,
    state_dict_key: str | None = None,
    strip_prefix: str = "",
    trace_input: Path | None = None,
    extra_files: list[Path] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Build and verify a package without assuming a domain, task, or architecture."""
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    rdf = yaml.safe_load(specification.read_text(encoding="utf-8"))
    if not isinstance(rdf, dict) or rdf.get("type") != "model":
        raise ValueError("specification must be a BioImage.IO model RDF mapping")
    output.mkdir(parents=True, exist_ok=True)
    _copy_declared_artifacts(rdf, specification.parent, output)

    weights = rdf.get("weights", {})
    if set(weights) != {"pytorch_state_dict"}:
        raise ValueError("prototype supports only pytorch_state_dict weights")
    state_descriptor = weights.get("pytorch_state_dict")
    if not isinstance(state_descriptor, dict):
        raise ValueError("specification requires weights.pytorch_state_dict")
    architecture = state_descriptor.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("pytorch_state_dict requires an architecture descriptor")
    architecture_source = architecture.get("source")
    constructor = _resolve_callable(
        architecture["callable"], output / architecture_source if architecture_source else None
    )
    model = constructor(**architecture.get("kwargs", {}))
    model.load_state_dict(
        _checkpoint_state(checkpoint, state_dict_key, strip_prefix), strict=True
    )
    model.eval()
    input_paths, expected_paths = _as_paths(test_input), _as_paths(test_output)
    if len(input_paths) != 1 or len(expected_paths) != 1:
        raise ValueError("prototype inference verification supports exactly one input and output")
    inference_input = torch.from_numpy(np.load(input_paths[0], allow_pickle=False))
    expected_output = np.load(expected_paths[0], allow_pickle=False)
    with torch.inference_mode():
        actual_output = model(inference_input).detach().cpu().numpy()
    if actual_output.shape != expected_output.shape or not np.allclose(
        actual_output, expected_output, rtol=1e-4, atol=1e-5
    ):
        raise ValueError(
            "strictly reloaded state dict does not reproduce the supplied test output"
        )
    weight_path = output / state_descriptor["source"]
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weight_path)  # nosec B614: tensors only

    for field, supplied in (("inputs", test_input), ("outputs", test_output)):
        descriptors, paths = _tensor_descriptors(rdf, field), _as_paths(supplied)
        if len(descriptors) != len(paths):
            raise ValueError(
                f"received {len(paths)} --test-{field[:-1]} values for "
                f"{len(descriptors)} RDF {field}"
            )
        for path, descriptor in zip(paths, descriptors):
            destination = output / descriptor["source"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    documentation = rdf.get("documentation")
    if not isinstance(documentation, dict) or not documentation.get("source"):
        raise ValueError("specification requires a local documentation source")
    documentation_path = output / documentation["source"]
    documentation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_card, documentation_path)
    for extra in extra_files or []:
        if not extra.is_file():
            raise FileNotFoundError(f"extra package file not found: {extra}")
        shutil.copy2(extra, output / extra.name)

    rdf["timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rdf_path = output / "rdf.yaml"
    _refresh_hashes(rdf, output)
    rdf_path.write_text(
        yaml.safe_dump(rdf, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    provenance_record = (
        json.loads(provenance.read_text(encoding="utf-8")) if provenance else {}
    )
    provenance_record.update(
        {
            "schema": "https://w3id.org/ro/crate/1.1",
            "created_utc": rdf["timestamp"],
            "model": {
                **provenance_record.get("model", {}),
                "name": rdf.get("name"),
                "version": rdf.get("version"),
            },
            "training": {
                **provenance_record.get("training", {}),
                "checkpoint_sha256": _digest(checkpoint),
            },
            "software": {
                **provenance_record.get("software", {}),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
            },
        }
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance_record, indent=2) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{_digest(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    verify(output)
    archive = output.with_suffix(".zip")
    _write_zip(output, archive)
    return output, archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-artifacts-dir", type=Path)
    parser.add_argument("--specification", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-input", type=Path, action="append")
    parser.add_argument("--test-output", type=Path, action="append")
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--state-dict-key")
    parser.add_argument("--strip-prefix", default="")
    parser.add_argument("--trace-input", type=Path)
    parser.add_argument(
        "--extra-file", type=Path, action="append", default=[],
        help="Additional file copied to the package root; may be repeated",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_artifacts_dir:
        directory = args.run_artifacts_dir
        conventional = {
            "specification": directory / "model-package.yaml",
            "checkpoint": directory / "weights.pt",
            "test_input": [directory / "test-input.npy"],
            "test_output": [directory / "test-output.npy"],
            "model_card": directory / "README.md",
            "provenance": directory / "run-provenance.json",
        }
        for name, value in conventional.items():
            if getattr(args, name) is not None:
                raise SystemExit(f"--run-artifacts-dir cannot be combined with --{name.replace('_', '-')}")
            setattr(args, name, value)
        args.extra_file.append(directory / "cli-parameters.json")
    missing = [name for name in ("specification", "checkpoint", "test_input", "test_output", "model_card")
               if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required packaging arguments: " + ", ".join(missing))
    output, archive = build_model_package(
        args.specification,
        args.checkpoint,
        args.test_input,
        args.test_output,
        args.model_card,
        args.output_dir,
        provenance=args.provenance,
        state_dict_key=args.state_dict_key,
        strip_prefix=args.strip_prefix,
        trace_input=args.trace_input,
        extra_files=args.extra_file,
        overwrite=args.overwrite,
    )
    print(f"Model package: {output}")
    print(f"BioImage.IO archive: {archive}")


if __name__ == "__main__":
    main()
