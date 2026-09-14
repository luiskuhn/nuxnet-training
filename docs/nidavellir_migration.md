# Testing the PyPI Nidavellir migration

This branch replaces the bundled `nidavellir_tools/` directory with
`nidavellir-tools==0.2.0` in both dependency manifests. Dataset table discovery,
ZIP extraction, and OME-TIFF reading come from the installed package. Downloading,
splitting, preprocessing, augmentation, model architecture, and training remain
in NuxNet. The existing NuxNet MC-dropout helper is unchanged.

## Release prerequisite

At preparation time, PyPI only offered `0.1.0`; it lacks `data_loading`.
Publish `0.2.0` containing the merged dataset-reader/uncertainty changes before
installing this branch or rebuilding its training image. Do not downgrade to
`0.1.0` or restore the bundled folder to work around an unavailable release.
If a different version is published, update both manifests together.

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
