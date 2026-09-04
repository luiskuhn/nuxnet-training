#!/usr/bin/env python3
"""Stage, inspect, load, validate, and publish portable model packages.

The package transport is intentionally project- and domain-independent.  A package is
either a directory containing ``rdf.yaml`` or a ZIP with ``rdf.yaml`` at its root.
Remote integrations are imported lazily so local staging/loading only requires the
dependencies already used by the project.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess  # nosec B404: only a discovered fixed executable is invoked
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml


class LoadedModelPackage(NamedTuple):
    """Executable model together with the metadata needed for another cycle."""

    model: torch.nn.Module
    rdf: dict[str, Any]
    provenance: dict[str, Any]
    root: Path
    representation: str


def _optional_module(name: str, extra: str) -> Any:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"{name} is required for this operation; install {extra}")
    return importlib.import_module(name)


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe ZIP member: {member.filename}")
        handle.extractall(destination)


def _write_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def _rdf_root(path: Path) -> Path:
    candidates = [path] if (path / "rdf.yaml").is_file() else [
        item.parent for item in path.rglob("rdf.yaml")
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one rdf.yaml below {path}, found {len(candidates)}")
    return candidates[0]


def _download_source(source: str, temporary: Path, revision: str | None) -> Path:
    if source.startswith("hf://"):
        hub = _optional_module("huggingface_hub", "huggingface_hub")
        return Path(hub.snapshot_download(repo_id=source[5:], revision=revision))
    if source.startswith("mlflow://"):
        mlflow = _optional_module("mlflow", "mlflow")
        # mlflow://<run-id>/<artifact-path> maps to MLflow's runs:/ URI.
        run_and_path = source[9:].split("/", 1)
        uri = f"runs:/{run_and_path[0]}/{run_and_path[1] if len(run_and_path) > 1 else ''}"
        return Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
    if source.startswith(("https://", "http://")):
        destination = temporary / "download"
        with urllib.request.urlopen(source) as response:  # nosec B310: explicit CLI input
            destination.write_bytes(response.read())
        return destination
    return Path(source).expanduser().resolve()


def stage(source: str, destination: Path, *, revision: str | None = None,
          overwrite: bool = False) -> Path:
    """Resolve a local/HTTP/MLflow/Hugging Face package into a stable directory."""
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"staging directory is not empty: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-stage-") as work:
        resolved = _download_source(source, Path(work), revision)
        if resolved.is_dir():
            package = _rdf_root(resolved)
            shutil.copytree(package, destination, dirs_exist_ok=True)
        elif zipfile.is_zipfile(resolved):
            extracted = Path(work) / "extracted"
            extracted.mkdir()
            _safe_extract(resolved, extracted)
            shutil.copytree(_rdf_root(extracted), destination, dirs_exist_ok=True)
        else:
            raise ValueError(f"source is not a model-package directory or ZIP: {source}")
    verify(destination)
    return destination


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(package: Path) -> dict[str, Any]:
    """Read the RDF and verify every local ``source`` that declares a SHA-256."""
    root = _rdf_root(package)
    rdf = yaml.safe_load((root / "rdf.yaml").read_text(encoding="utf-8"))
    if not isinstance(rdf, dict) or rdf.get("type") != "model":
        raise ValueError("rdf.yaml does not describe a BioImage.IO model")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            source, expected = value.get("source"), value.get("sha256")
            if isinstance(source, str) and isinstance(expected, str):
                artifact = (root / source).resolve()
                if not artifact.is_relative_to(root.resolve()) or not artifact.is_file():
                    raise ValueError(f"missing or unsafe RDF artifact: {source}")
                if _digest(artifact) != expected:
                    raise ValueError(f"checksum mismatch for {source}")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(rdf)
    return rdf


def _resolve_callable(specification: str, source: Path | None = None) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if source is not None and not separator:
        module_name, attribute = f"_portable_model_{_digest(source)[:12]}", specification
        module_spec = importlib.util.spec_from_file_location(module_name, source)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot import architecture source: {source}")
        target = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(target)
    elif not separator:
        raise ValueError(f"architecture callable must be module:attribute, got {specification}")
    else:
        target = importlib.import_module(module_name)
    for component in attribute.split("."):
        target = getattr(target, component)
    return target


def load_package(
    package: Path, *, representation: str = "auto"
) -> LoadedModelPackage:
    """Load an executable and retain its RDF, provenance, and representation."""
    root = _rdf_root(package)
    rdf = verify(root)
    weights = rdf.get("weights", {})
    if representation in {"auto", "torchscript"} and "torchscript" in weights:
        model = torch.jit.load(str(root / weights["torchscript"]["source"]), map_location="cpu")
        selected = "torchscript"
    elif representation not in {"auto", "pytorch_state_dict"}:
        raise ValueError(f"unsupported representation: {representation}")
    else:
        descriptor = weights.get("pytorch_state_dict")
        if not descriptor:
            raise ValueError("package has no supported PyTorch representation")
        architecture = descriptor["architecture"]
        source = architecture.get("source")
        constructor = _resolve_callable(
            architecture["callable"], root / source if source else None
        )
        model = constructor(**architecture.get("kwargs", {}))
        state = torch.load(
            root / descriptor["source"], map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        selected = "pytorch_state_dict"
    provenance_path = root / "provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    return LoadedModelPackage(model.eval(), rdf, provenance, root, selected)


def load(package: Path, *, representation: str = "auto") -> torch.nn.Module:
    """Compatibility helper returning only the executable from ``load_package``."""
    return load_package(package, representation=representation).model


def _checkpoint_state(
    checkpoint: Path, state_dict_key: str | None, strip_prefix: str
) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state_dict_key:
        if not isinstance(loaded, dict) or state_dict_key not in loaded:
            raise ValueError(f"checkpoint has no {state_dict_key!r} mapping")
        loaded = loaded[state_dict_key]
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in loaded.items()
    ):
        raise ValueError("checkpoint must resolve to a tensor state dictionary")
    if strip_prefix:
        loaded = {
            key.removeprefix(strip_prefix): value
            for key, value in loaded.items()
            if key.startswith(strip_prefix)
        }
        if not loaded:
            raise ValueError(f"no checkpoint tensors start with {strip_prefix!r}")
    return loaded


def export_child_package(
    parent: Path,
    checkpoint: Path,
    test_output: Path,
    model_card: Path,
    destination: Path,
    *,
    version: str,
    name: str | None = None,
    description: str | None = None,
    parent_identifier: str | None = None,
    state_dict_key: str | None = None,
    strip_prefix: str = "",
    overwrite: bool = False,
) -> Path:
    """Export trained weights using a parent's executable and metadata contract.

    The caller supplies a newly evaluated output fixture and model card so stale
    technical or scientific claims can never silently flow into the child.
    """
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {destination}")
        shutil.rmtree(destination)
    parent_root = _rdf_root(parent)
    parent_rdf = verify(parent_root)
    descriptor = parent_rdf.get("weights", {}).get("pytorch_state_dict")
    if not descriptor:
        raise ValueError("parent requires pytorch_state_dict weights for retraining")
    shutil.copytree(parent_root, destination)
    for kind, old_descriptor in parent_rdf["weights"].items():
        if kind != "pytorch_state_dict" and isinstance(old_descriptor, dict):
            stale_source = old_descriptor.get("source")
            if stale_source:
                (destination / stale_source).unlink(missing_ok=True)
    state = _checkpoint_state(checkpoint, state_dict_key, strip_prefix)

    # Strict reconstruction proves that the child remains architecture-compatible.
    architecture = descriptor["architecture"]
    architecture_source = architecture.get("source")
    constructor = _resolve_callable(
        architecture["callable"],
        destination / architecture_source if architecture_source else None,
    )
    model = constructor(**architecture.get("kwargs", {}))
    model.load_state_dict(state, strict=True)

    weight_path = destination / descriptor["source"]
    torch.save(model.state_dict(), weight_path)  # nosec B614: tensors only
    shutil.copy2(test_output, destination / parent_rdf["outputs"][0]["test_tensor"]["source"])
    shutil.copy2(model_card, destination / parent_rdf["documentation"]["source"])

    rdf = yaml.safe_load(yaml.safe_dump(parent_rdf))
    rdf["version"] = version
    rdf["timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if name:
        rdf["name"] = name
    if description:
        rdf["description"] = description
    rdf["weights"] = {"pytorch_state_dict": descriptor}
    rdf["weights"]["pytorch_state_dict"]["sha256"] = _digest(weight_path)
    output_descriptor = rdf["outputs"][0]["test_tensor"]
    output_descriptor["sha256"] = _digest(destination / output_descriptor["source"])
    documentation = rdf["documentation"]
    documentation["sha256"] = _digest(destination / documentation["source"])

    old_provenance = destination / "provenance.json"
    parent_provenance = (
        json.loads(old_provenance.read_text(encoding="utf-8"))
        if old_provenance.is_file()
        else {}
    )
    provenance = {
        "schema": "https://w3id.org/ro/crate/1.1",
        "created_utc": rdf["timestamp"],
        "model": {"name": rdf.get("name"), "version": version},
        "parent": {
            "identifier": parent_identifier or str(parent_root),
            "name": parent_rdf.get("name"),
            "version": parent_rdf.get("version"),
            "provenance": parent_provenance,
        },
        "training": {"checkpoint_sha256": _digest(checkpoint)},
    }
    old_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (destination / "rdf.yaml").write_text(
        yaml.safe_dump(rdf, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    checksum_manifest = destination / "SHA256SUMS"
    if checksum_manifest.exists():
        checksums = [
            f"{_digest(path)}  {path.relative_to(destination)}"
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path != checksum_manifest
        ]
        checksum_manifest.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    verify(destination)
    _write_zip(destination, destination.with_suffix(".zip"))
    return destination


def publish_huggingface(package: Path, repo_id: str, *, private: bool = False,
                        revision: str = "main", commit_message: str | None = None) -> str:
    """Validate and upload a package directory to a Hugging Face model repository."""
    root = _rdf_root(package)
    verify(root)
    hub = _optional_module("huggingface_hub", "huggingface_hub")
    api = hub.HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(root), revision=revision,
        commit_message=commit_message or "Publish FAIR model package",
    )
    return str(result)


def validate_bioimageio(package: Path) -> None:
    """Run the official BioImage.IO CLI against a package directory or archive."""
    executable = shutil.which("bioimageio")
    if executable is None:
        raise RuntimeError("bioimageio is required; install bioimageio.core")
    subprocess.run([executable, "test", str(package)], check=True)  # nosec B603


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("source", help="path, URL, hf://repo-id, or mlflow://run-id/path")
    stage_parser.add_argument("destination", type=Path)
    stage_parser.add_argument("--revision")
    stage_parser.add_argument("--overwrite", action="store_true")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("package", type=Path)
    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("package", type=Path)
    load_parser.add_argument("--representation", default="auto",
                             choices=["auto", "torchscript", "pytorch_state_dict"])
    load_parser.add_argument(
        "--weights-output", type=Path, help="Write tensor-only initialization weights"
    )
    load_parser.add_argument(
        "--metadata-output", type=Path, help="Write RDF, provenance, and selection as JSON"
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("package", type=Path)
    publish_parser = subparsers.add_parser("publish-hf")
    publish_parser.add_argument("package", type=Path)
    publish_parser.add_argument("repo_id")
    publish_parser.add_argument("--private", action="store_true")
    publish_parser.add_argument("--revision", default="main")
    publish_parser.add_argument("--commit-message")
    child_parser = subparsers.add_parser("export-child")
    child_parser.add_argument("parent", type=Path)
    child_parser.add_argument("checkpoint", type=Path)
    child_parser.add_argument("destination", type=Path)
    child_parser.add_argument("--test-output", type=Path, required=True)
    child_parser.add_argument("--model-card", type=Path, required=True)
    child_parser.add_argument("--version", required=True)
    child_parser.add_argument("--name")
    child_parser.add_argument("--description")
    child_parser.add_argument("--parent-identifier")
    child_parser.add_argument("--state-dict-key")
    child_parser.add_argument("--strip-prefix", default="")
    child_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "stage":
        print(
            stage(
                args.source,
                args.destination,
                revision=args.revision,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "inspect":
        print(json.dumps(verify(args.package), indent=2, default=str))
    elif args.command == "load":
        loaded = load_package(args.package, representation=args.representation)
        if args.weights_output:
            if loaded.representation != "pytorch_state_dict":
                raise ValueError(
                    "--weights-output requires --representation pytorch_state_dict"
                )
            args.weights_output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = loaded.rdf["weights"]["pytorch_state_dict"]
            shutil.copy2(loaded.root / descriptor["source"], args.weights_output)
        metadata = {
            "representation": loaded.representation,
            "weights_sha256": (
                loaded.rdf["weights"][loaded.representation].get("sha256")
            ),
            "rdf": loaded.rdf,
            "provenance": loaded.provenance,
        }
        if args.metadata_output:
            args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_output.write_text(
                json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
            )
        print(json.dumps(metadata, indent=2, default=str))
    elif args.command == "validate":
        verify(args.package)
        validate_bioimageio(args.package)
    elif args.command == "publish-hf":
        print(publish_huggingface(args.package, args.repo_id, private=args.private,
                                  revision=args.revision, commit_message=args.commit_message))
    elif args.command == "export-child":
        print(
            export_child_package(
                args.parent,
                args.checkpoint,
                args.test_output,
                args.model_card,
                args.destination,
                version=args.version,
                name=args.name,
                description=args.description,
                parent_identifier=args.parent_identifier,
                state_dict_key=args.state_dict_key,
                strip_prefix=args.strip_prefix,
                overwrite=args.overwrite,
            )
        )


if __name__ == "__main__":
    main()
