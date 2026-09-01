"""Reproducibility and MLflow provenance helpers inspired by mlf-core."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch


class MLFCore:
    @staticmethod
    def set_general_random_seeds(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def set_pytorch_random_seeds(seed, num_gpus=None):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def log_runtime_environment():
        """Log portable runtime/package provenance without requiring Conda."""
        report = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()},
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        output = Path(tempfile.mkdtemp()) / "runtime_environment.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if mlflow.active_run():
            mlflow.log_artifact(str(output), artifact_path="reports")

    @staticmethod
    def md5(filename):
        digest = hashlib.md5()  # nosec B324
        with open(filename, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def get_md5_sums(cls, directory):
        path = Path(directory)
        files = [path] if path.is_file() else [candidate for candidate in sorted(path.rglob("*")) if candidate.is_file()]
        return [(str(candidate), cls.md5(candidate)) for candidate in files]

    @classmethod
    def log_input_data(cls, input_data):
        hashes = cls.get_md5_sums(input_data)
        manifest = Path(tempfile.mkdtemp()) / "input_checksums.json"
        manifest.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        if mlflow.active_run():
            mlflow.log_artifact(str(manifest), artifact_path="reports")
