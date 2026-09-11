#!/usr/bin/env python3
"""Regenerate RDF-declared sample TIFFs from staged exact test tensors.

This command is for an existing ``model-package-inputs`` directory.  It neither
loads a model nor retrains it: the RDF identifies the source NPY tensors and the
destination TIFF files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nidavellir_tools.sample_tensors import (  # noqa: E402
    CHANNEL_POLICIES,
    LAYOUTS,
    create_sample_tensor,
)


def create_samples(directory: Path, layout: str, batch_index: int, channel_policy: str):
    """Recreate every RDF-declared input and output sample in ``directory``."""
    rdf_path = directory / "model-package.yaml"
    if not rdf_path.is_file():
        raise FileNotFoundError(f"staged RDF not found: {rdf_path}")
    rdf = yaml.safe_load(rdf_path.read_text(encoding="utf-8"))
    records = []
    # Inputs and outputs are handled identically and may resolve different axes.
    for field in ("inputs", "outputs"):
        tensors = rdf.get(field)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"RDF requires at least one {field} tensor")
        for tensor in tensors:
            test = tensor.get("test_tensor")
            sample = tensor.get("sample_tensor")
            if not isinstance(test, dict) or not test.get("source"):
                raise ValueError(f"every RDF {field} tensor requires test_tensor.source")
            if not isinstance(sample, dict) or not sample.get("source"):
                raise ValueError(f"every RDF {field} tensor requires sample_tensor.source")
            source = directory / test["source"]
            if not source.is_file():
                raise FileNotFoundError(f"test tensor not found: {source}")
            records.append(
                create_sample_tensor(
                    np.load(source, allow_pickle=False),
                    tensor.get("axes"),
                    directory / sample["source"],
                    layout=layout,
                    batch_index=batch_index,
                    channel_policy=channel_policy,
                )
            )
    # Re-running the command replaces only the sample-generation portion of the
    # provenance record, leaving training and lineage information intact.
    provenance_path = directory / "run-provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    provenance["sample_tensors"] = records
    provenance_path.write_text(
        json.dumps(provenance, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-artifacts-dir", required=True, type=Path)
    parser.add_argument("--sample-layout", choices=LAYOUTS, default="auto")
    parser.add_argument("--sample-batch-index", type=int, default=0)
    parser.add_argument(
        "--sample-channel-policy",
        choices=CHANNEL_POLICIES,
        default="squeeze-singleton",
    )
    return parser


def main():
    args = build_parser().parse_args()
    create_samples(
        args.run_artifacts_dir,
        args.sample_layout,
        args.sample_batch_index,
        args.sample_channel_policy,
    )


if __name__ == "__main__":
    main()
