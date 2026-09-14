# Testing the PyPI Nidavellir migration

This branch replaces the bundled `nidavellir_tools/` directory with
`nidavellir-tools==0.2.0` in both dependency manifests. Dataset table discovery,
ZIP extraction, and OME-TIFF reading come from the installed package. Downloading,
splitting, preprocessing, augmentation, model architecture, and training remain
in NuxNet. The existing NuxNet MC-dropout helper is unchanged.

## Release prerequisite

PyPI `0.2.0` is published and was installed successfully for the validation below.
Do not downgrade to `0.1.0`: it lacks `data_loading`. If an index temporarily lists
only the old release, retry with `--no-cache-dir --index-url https://pypi.org/simple`.
Do not restore the bundled folder. Update both manifests together for future versions.

## Verified CPU integration (2026-09-14)

- Fresh Docker `python:3.12-slim` container on Linux ARM64, using PyPI
  `nidavellir-tools==0.2.0` and this repository's pinned requirements.
- `pip check`: no broken requirements.
- Dataset-reader and uncertainty imports resolve to installed `site-packages`;
  `nidavellir --help` succeeds.
- Full suite: **82 passed, 1 failed** in 24.40 seconds. The failure is the
  previously observed `test_new_cli_hyperparameters_are_serializable_and_range_checked`:
  `PosixPath('/workspace/model-package.yaml')` is not JSON serializable.
- Reader identity/resources tests, existing dataset tests, packaging tests, and
  two-process CPU distributed tests passed. No local development wheel was used.
- The NVIDIA CUDA/AMD64 production image, real-data GPU training, and full
  parent-to-child transfer-learning workflow remain to be validated on the VM.

## On the GPU VM

From your existing NuxNet checkout, preserve any local changes before switching:

```bash
git fetch origin
git switch --track origin/feature/pypi-nidavellir-tools
```

If the branch already exists locally, use `git switch feature/pypi-nidavellir-tools`
and `git pull --ff-only` instead.

### Python environment

Activate your intended Python 3.12 training environment first:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c 'from importlib.metadata import version; from nidavellir_tools import data_loading; print(version("nidavellir-tools")); print(data_loading.__file__)'
python -m pip install pytest
python -m pytest -q tests/test_nidavellir_integration.py tests/test_data_loader.py tests/test_spacing_rotation_loss.py
python -m pytest -q
```

The reported version should be `0.2.0` and the reader path should be inside the
environment, not inside this repository. Use `nidavellir --help` to check CLI
installation. A previously observed failure in
`test_new_cli_hyperparameters_are_serializable_and_range_checked` concerns JSON
serialization of a `Path` CLI value; it predates this extraction and must not be
mistaken for a loader mismatch.

### Rebuild the training container

The existing Dockerfile already installs `requirements.txt`. On the NVIDIA GPU
VM, build a separate image tag so the previous image remains available:

```bash
docker build -t nuxnet-training:pypi-nidavellir .
docker run --rm --entrypoint python nuxnet-training:pypi-nidavellir -m pip check
docker run --rm --entrypoint python nuxnet-training:pypi-nidavellir -c 'from importlib.metadata import version; from nidavellir_tools import data_loading; print(version("nidavellir-tools")); print(data_loading.__file__)'
docker run --rm --gpus all --entrypoint python nuxnet-training:pypi-nidavellir -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); assert torch.cuda.is_available()'
```

Use your existing training command, dataset mounts, and configuration, changing
only the image tag and choosing fresh run/output directories. Start with a short
training run before a full experiment. Do not change dataset metadata or
preprocessing for this migration. Inspect losses, sample shapes, calibration,
checkpoints, and generated model-package inputs.

After training succeeds, follow the existing parent/child workflow in the root
README using this same image tag. Its package commands now use the installed
`nidavellir` entry point. Validation and remote publishing may require optional
extras; no credentials or remote writes are needed for local reader tests.

GPU training and the full transfer-learning workflow are follow-up validation,
not implied by a passing CPU test suite.
