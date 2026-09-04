#!/usr/bin/env python3
"""Export a trained NuxNet model as a FAIR, repository-ready model package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numorph_nuclei_segmentation.model import UNet3D  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,  # nosec B603 B607
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a PyTorch state dictionary")
    # Accept either the exported UNet weights or a Lightning NumorphSegmentator checkpoint.
    if any(key.startswith("model.") for key in state):
        state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    return state


def _write_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def validate_args(args: argparse.Namespace) -> None:
    """Validate package metadata and tensor options before doing expensive work."""
    if not (args.doi or args.citation_url):
        raise ValueError(
            "provide --doi or --citation-url so the citation is resolvable"
        )
    if len(args.test_shape) != 3 or any(
        size < 4 or size % 4 for size in args.test_shape
    ):
        raise ValueError(
            "--test-shape must contain three comma-separated dimensions divisible by four"
        )
    if args.input_channels < 1 or args.classes < 1:
        raise ValueError("--input-channels and --classes must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in the interval [0, 1)")
    target_spacing = tuple(getattr(args, "target_voxel_spacing", (3.0, 1.0, 1.0)))
    if len(target_spacing) != 3 or any(value <= 0 for value in target_spacing):
        raise ValueError(
            "--target-voxel-spacing must contain three positive values in Z,Y,X order"
        )
    if args.cover and Path(args.cover).suffix.lower() not in {
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".svg",
    }:
        raise ValueError("--cover must be a GIF, JPEG, PNG, or SVG image")


def export_model(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output}; pass --overwrite to replace it"
            )
        shutil.rmtree(output)
    validate_args(args)
    output.mkdir(parents=True, exist_ok=True)

    model = UNet3D(
        in_channels=args.input_channels, classes=args.classes, dropout=args.dropout
    )
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    model.eval()

    weights = output / "weights.pt"
    torch.save(model.state_dict(), weights)  # nosec B614: tensor-only state dictionary
    generator = torch.Generator().manual_seed(0)
    raw_example = torch.rand(
        (1, args.input_channels, *args.test_shape),
        generator=generator,
        dtype=torch.float32,
    )
    if args.normalize_input:
        example = (raw_example - raw_example.amin(dim=(2, 3, 4), keepdim=True)) / (
            raw_example.amax(dim=(2, 3, 4), keepdim=True)
            - raw_example.amin(dim=(2, 3, 4), keepdim=True)
        )
    else:
        example = raw_example
    with torch.inference_mode():
        prediction = model(example).softmax(dim=1)
        traced = torch.jit.trace(model, example)
    traced.save(str(output / "model.ts"))
    np.save(output / "test-input.npy", raw_example.numpy())
    np.save(output / "test-output.npy", prediction.numpy())
    shutil.copy2(
        ROOT / "numorph_nuclei_segmentation/model/unet_3d_models.py",
        output / "unet_3d_models.py",
    )
    shutil.copy2(ROOT / "environment.yml", output / "environment.yml")
    shutil.copy2(ROOT / "LICENSE", output / "LICENSE")
    cover_source = (
        Path(args.cover).resolve()
        if args.cover
        else ROOT / "docs/images/graph_abstract_nuxnet_training.png"
    )
    if not cover_source.is_file():
        raise FileNotFoundError(f"cover image not found: {cover_source}")
    shutil.copy2(cover_source, output / f"cover{cover_source.suffix.lower()}")

    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    commit = args.git_commit or git_value("rev-parse", "HEAD")
    remote = args.source_repository or git_value("config", "--get", "remote.origin.url")
    authors = [{"name": args.author, "github_user": args.github_user}]
    if args.author_orcid:
        authors[0]["orcid"] = args.author_orcid
    target_spacing = tuple(getattr(args, "target_voxel_spacing", (3.0, 1.0, 1.0)))
    provenance = {
        "schema": "https://w3id.org/ro/crate/1.1",
        "created_utc": created,
        "model": {
            "name": args.name,
            "version": args.model_version,
            "checkpoint_sha256": sha256(checkpoint),
            "exported_weights_sha256": sha256(weights),
        },
        "training": {
            "dataset": args.dataset,
            "dataset_version": args.dataset_version,
            "mlflow_run_id": args.mlflow_run_id,
            "normalize_input": args.normalize_input,
            "target_voxel_spacing_zyx_um": target_spacing,
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "source": {"repository": remote, "git_commit": commit},
        "export_command": sys.argv,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    card = f"""---
license: {json.dumps(args.license)}
library_name: pytorch
pipeline_tag: image-segmentation
tags:
- bioimageio
- 3d
- unet
datasets:
- {json.dumps(args.dataset)}
---

# {args.name}

{args.description}

## Model details

- **Version:** {args.model_version}
- **Developed and maintained by:** {args.author} (`@{args.github_user}`)
- **Model type:** Residual 3D U-Net with two downsampling levels
- **Input modality:** Single-channel light-sheet fluorescence microscopy
- **License:** {args.license}
- **Source:** {remote or "Not recorded"} at revision `{commit or "not recorded"}`

## Uses

The direct use is detection of manually curated middle-Z nucleus markers in cleared mouse-brain microscopy. The model can be integrated into a patch-based inference pipeline or fine-tuned on compatible annotated volumes. It is out of scope for clinical/diagnostic use, complete 3D boundary segmentation, instance segmentation, or unvalidated imaging modalities.

## Task details

Input is `float32` in `BCZYX` order on a `{target_spacing}` µm/voxel Z,Y,X grid. Spatial dimensions must be divisible by four. Source OME volumes must be spacing-normalized before invoking the tensor model. {"BioImage.IO consumers apply per-volume min/max scaling over `Z`, `Y`, and `X`." if args.normalize_input else "No intensity normalization is included; inputs must already match the values used for training."} The network emits `{args.classes}`-channel logits; the packaged BioImage.IO postprocessing applies softmax and returns probabilities in the same spatial shape.

## Bias, risks, and limitations

Training annotations are sparse two-dimensional middle-Z markers embedded in volumes, not complete nucleus boundaries. Performance may shift with species, stain, microscope, clearing protocol, voxel spacing, signal-to-noise ratio, or density. Validate against representative held-out data, inspect errors manually, and do not interpret marker probabilities as biological measurements without appropriate controls.

## Training details

- **Dataset:** `{args.dataset}` (version `{args.dataset_version}`)
- **MLflow run:** `{args.mlflow_run_id or "Not recorded"}`
- **Objective:** focal loss (gamma 2) with Adam optimization
- **Augmentation:** foreground-aware random crops and small random 3D rotations
- **Environment:** [`environment.yml`](environment.yml)
- **Machine-readable provenance:** [`provenance.json`](provenance.json)

## Validation

Technical reproducibility is represented by `test-input.npy` and `test-output.npy`; run `bioimageio test rdf.yaml` before publication. These tensors test execution, not scientific accuracy. Add the held-out IoU, sample counts, fold, acquisition strata, and external-validation results for this specific run here before publishing.

## Technical specifications and files

`model.ts` is the portable TorchScript model and emits raw logits. `weights.pt` is the tensor-only PyTorch state dictionary; `unet_3d_models.py` reconstructs its architecture. `rdf.yaml` defines the BioImage.IO execution pipeline, including normalization and softmax. See `ARTIFACTS.md` for the complete inventory and repository upload mapping.

## Citation

{args.citation}{f" DOI: {args.doi}." if args.doi else f" {args.citation_url}"}
"""
    (output / "README.md").write_text(card, encoding="utf-8")

    cover = next(path for path in output.iterdir() if path.name.startswith("cover."))
    citation = {"text": args.citation}
    citation["doi" if args.doi else "url"] = args.doi or args.citation_url
    spatial_input_axes = [
        {"id": axis, "type": "space", "size": {"min": 4, "step": 4}}
        for axis in ("z", "y", "x")
    ]
    spatial_output_axes = [
        {"id": axis, "type": "space", "size": {"tensor_id": "input", "axis_id": axis}}
        for axis in ("z", "y", "x")
    ]
    preprocessing = [{"id": "ensure_dtype", "kwargs": {"dtype": "float32"}}]
    if args.normalize_input:
        preprocessing.append(
            {
                "id": "scale_range",
                "kwargs": {
                    "axes": ["z", "y", "x"],
                    "min_percentile": 0.0,
                    "max_percentile": 100.0,
                },
            }
        )
    rdf = {
        "format_version": "0.5.14",
        "type": "model",
        "timestamp": created,
        "name": args.name,
        "version": args.model_version,
        "description": args.description,
        "authors": authors,
        "maintainers": [{"name": args.author, "github_user": args.github_user}],
        "license": args.license,
        "documentation": {
            "source": "README.md",
            "sha256": sha256(output / "README.md"),
        },
        "covers": [{"source": cover.name, "sha256": sha256(cover)}],
        "git_repo": remote,
        "tags": [
            "3d",
            "pytorch",
            "unet",
            "nucleus",
            "segmentation",
            "light-sheet-microscopy",
            "mouse",
        ],
        "cite": [citation],
        "inputs": [
            {
                "id": "input",
                "description": "Raw microscopy intensity volume",
                "axes": [
                    {"type": "batch", "size": 1},
                    {
                        "type": "channel",
                        "channel_names": [
                            f"channel_{i}" for i in range(args.input_channels)
                        ],
                    },
                    *spatial_input_axes,
                ],
                "data": {"type": "float32", "range": [None, None]},
                "preprocessing": preprocessing,
                "test_tensor": {
                    "source": "test-input.npy",
                    "sha256": sha256(output / "test-input.npy"),
                },
            }
        ],
        "outputs": [
            {
                "id": "probabilities",
                "description": "Background and nucleus-marker probabilities",
                "axes": [
                    {"type": "batch"},
                    {
                        "type": "channel",
                        "channel_names": ["background", "marker"]
                        if args.classes == 2
                        else [f"class_{i}" for i in range(args.classes)],
                    },
                    *spatial_output_axes,
                ],
                "data": {"type": "float32", "range": [0.0, 1.0]},
                "postprocessing": [
                    {"id": "softmax", "kwargs": {"axis": "channel"}},
                    {"id": "ensure_dtype", "kwargs": {"dtype": "float32"}},
                ],
                "test_tensor": {
                    "source": "test-output.npy",
                    "sha256": sha256(output / "test-output.npy"),
                },
            }
        ],
        "weights": {
            "torchscript": {
                "source": "model.ts",
                "sha256": sha256(output / "model.ts"),
                "parent": "pytorch_state_dict",
                "pytorch_version": str(torch.__version__),
            },
            "pytorch_state_dict": {
                "source": "weights.pt",
                "sha256": sha256(weights),
                "architecture": {
                    "source": "unet_3d_models.py",
                    "sha256": sha256(output / "unet_3d_models.py"),
                    "callable": "UNet3D",
                    "kwargs": {
                        "in_channels": args.input_channels,
                        "classes": args.classes,
                        "dropout": args.dropout,
                    },
                },
                "dependencies": {
                    "source": "environment.yml",
                    "sha256": sha256(output / "environment.yml"),
                },
                "pytorch_version": str(torch.__version__),
            },
        },
        "config": {
            "nuxnet": {
                "provenance": "provenance.json",
                "repository": remote,
                "git_commit": commit,
                "target_voxel_spacing_zyx_um": list(target_spacing),
            }
        },
    }
    (output / "rdf.yaml").write_text(
        yaml.safe_dump(rdf, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    inventory = """# Exported artifact inventory

| Artifact | BioImage.IO | Hugging Face | Purpose |
| --- | --- | --- | --- |
| `rdf.yaml` | Required entry point | Useful supplementary metadata | BioImage.IO 0.5.14 execution and metadata contract. |
| `model.ts` | Primary portable weights | Downloadable model file | TorchScript network; returns raw logits. |
| `weights.pt` | Alternative weights | Primary PyTorch weights | Tensor-only `UNet3D` state dictionary. |
| `unet_3d_models.py` | Required by `weights.pt` | Loading implementation | Exact architecture source. |
| `test-input.npy` / `test-output.npy` | Required reproducibility pair | Technical smoke-test fixtures | Raw input and expected post-softmax output. |
| `cover.<ext>` | Required representative cover | Repository preview asset | Visual summary shown in model discovery. |
| `README.md` | Required documentation | Required model card | Uses, limitations, training, validation, and YAML Hub metadata. |
| `environment.yml` | State-dict dependency | Reproducible environment | Pinned model runtime dependencies. |
| `provenance.json` | Additional file | Provenance record | Dataset/run/source/software/checksum lineage. |
| `LICENSE` | Distribution terms | Distribution terms | Model-package license text. |
| `SHA256SUMS` | Integrity supplement | Integrity supplement | SHA-256 digest of every file above. |

Upload the sibling ZIP as one BioImage.IO resource. Upload the directory contents as one Hugging Face model repository; Hugging Face reads the front matter from `README.md`. `rdf.yaml` references only files inside the ZIP and records hashes for executable inputs, outputs, architecture, environment, documentation, cover, and weights.
"""
    (output / "ARTIFACTS.md").write_text(inventory, encoding="utf-8")
    checksums = [
        f"{sha256(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file()
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    archive = output.with_suffix(".zip")
    _write_zip(output, archive)
    return output, archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Exported .pt state dictionary or Lightning checkpoint",
    )
    parser.add_argument("--output-dir", default="output/model-package")
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--author-orcid")
    parser.add_argument(
        "--github-user", required=True, help="GitHub username of the model maintainer"
    )
    parser.add_argument("--license", default="MIT")
    parser.add_argument("--citation", required=True)
    parser.add_argument("--doi")
    parser.add_argument("--citation-url")
    parser.add_argument(
        "--dataset", required=True, help="Resolvable dataset name or identifier"
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--mlflow-run-id")
    parser.add_argument("--source-repository")
    parser.add_argument("--git-commit")
    parser.add_argument(
        "--cover",
        help="Representative PNG/JPEG/GIF/SVG; defaults to the NuxNet graphical abstract",
    )
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--normalize-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the normalization setting used for training",
    )
    parser.add_argument(
        "--target-voxel-spacing",
        type=lambda value: tuple(
            float(component.strip()) for component in value.split(",")
        ),
        default=(3.0, 1.0, 1.0),
        metavar="Z,Y,X",
        help="Required input grid in Z,Y,X order, in µm/voxel",
    )
    parser.add_argument(
        "--test-shape",
        type=lambda value: tuple(map(int, value.split(","))),
        default=(4, 16, 16),
        metavar="Z,Y,X",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run `bioimageio test` on the completed ZIP (requires bioimageio.core)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output, archive = export_model(args)
    if args.validate:
        executable = shutil.which("bioimageio")
        if executable is None:
            raise RuntimeError(
                "--validate requires the `bioimageio` command from bioimageio.core"
            )
        subprocess.run([executable, "test", str(archive)], check=True)  # nosec B603
    print(f"Model package: {output}")
    print(f"BioImage.IO archive: {archive}")


if __name__ == "__main__":
    main()
