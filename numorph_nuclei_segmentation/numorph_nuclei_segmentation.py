"""NuxNet training entry point."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print

from numorph_nuclei_segmentation.data_loading.data_loader import (
    DEFAULT_TARGET_VOXEL_SIZE_UM,
    NUMORPH_DATASET_URL,
    NumorphDataModule,
)
from numorph_nuclei_segmentation.mlf_core.mlf_core import MLFCore
from numorph_nuclei_segmentation.model.model import NumorphSegmentator


def parse_target_voxel_size(value: str) -> tuple[float, float, float]:
    """Parse positive physical voxel sizes in ``Z,Y,X`` order."""
    try:
        voxel_size_um = tuple(
            float(component.strip()) for component in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "target voxel size must contain three numeric values in Z,Y,X order"
        ) from error
    if len(voxel_size_um) != 3 or any(component <= 0 for component in voxel_size_um):
        raise argparse.ArgumentTypeError(
            "target voxel size must contain three positive values in Z,Y,X order"
        )
    return voxel_size_um


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_integer(value: str) -> int:
    """Parse an integer that can safely be used as an epoch frequency."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("probability must be between 0 and 1")
    return parsed


def overlap(value: str) -> float:
    """Parse a sliding-window overlap fraction in the half-open range [0, 1)."""
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("overlap must be in [0, 1)")
    return parsed


def optional_path(value: str) -> str | None:
    """Translate MLproject's explicit empty-path sentinel."""
    return None if value.strip().lower() in {"", "none"} else value


def validate_parent_initialization(weights: str | None, metadata: str | None) -> None:
    """Ensure transfer weights and their exported metadata remain paired."""
    if bool(weights) != bool(metadata):
        raise ValueError(
            "--initial-weights and --parent-metadata must be provided together"
        )
    if not weights:
        return
    weights_path, metadata_path = Path(weights), Path(metadata)
    if not weights_path.is_file():
        raise FileNotFoundError(f"initial weights not found: {weights_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"parent metadata not found: {metadata_path}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = record.get("weights_sha256")
    checksum = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if not expected or digest != expected:
        raise ValueError("initial weights do not match the parent metadata checksum")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train NuxNet on BIA-described OME-TIFF volumes"
    )
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument(
        "--devices", default="auto", help="Lightning device count or 'auto'"
    )
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--general-seed", type=int, default=0)
    parser.add_argument("--pytorch-seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Factor applied when foreground validation IoU plateaus",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=5,
        help="Validation evaluations without sufficient IoU improvement before LR reduction",
    )
    parser.add_argument(
        "--lr-scheduler-threshold",
        type=float,
        default=1e-5,
        help="Minimum absolute foreground validation-IoU improvement",
    )
    parser.add_argument(
        "--lr-scheduler-cooldown",
        type=int,
        default=2,
        help="Validation evaluations to wait after a learning-rate reduction",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate used by the plateau scheduler",
    )
    parser.add_argument("--training-batch-size", type=int, default=1)
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Number of validation/test inference windows evaluated simultaneously",
    )
    parser.add_argument(
        "--inference-overlap",
        type=overlap,
        default=0.5,
        help="Fractional overlap between validation/test sliding windows in [0,1)",
    )
    parser.add_argument(
        "--class-weights",
        default="1.0,1.0",
        help="Explicit per-class weights (default is unweighted)",
    )
    parser.add_argument("--ce-loss-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=nonnegative_float, default=1.0)
    parser.add_argument(
        "--cross-validation-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--validation-fold",
        type=int,
        default=0,
        help="One-based fold used for validation and test metrics; 0 selects a fold from --general-seed",
    )
    parser.add_argument(
        "--test-epochs",
        type=positive_integer,
        default=10,
        help=(
            "Epochs between validation runs and plateau-scheduler observations; "
            "must be positive"
        ),
    )
    parser.add_argument("--dataset-path", default="/data")
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download the dataset to --dataset-path if needed",
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="Replace an existing downloaded dataset",
    )
    parser.add_argument(
        "--dataset-url",
        default=NUMORPH_DATASET_URL,
        help="Public HTTP(S) or Google Drive dataset URL",
    )
    parser.add_argument("--n-channels", type=int, default=1)
    parser.add_argument("--n-class", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument(
        "--initial-weights",
        type=optional_path,
        help="Tensor-only state dictionary used to initialize a new training cycle",
    )
    parser.add_argument(
        "--parent-metadata",
        type=optional_path,
        help="JSON metadata emitted with --initial-weights by nidavellir_tools/model_package_registry.py",
    )
    parser.add_argument(
        "--patch-size",
        default="32,128,128",
        help="Training patch dimensions Z,Y,X; each must be divisible by 4",
    )
    parser.add_argument(
        "--target-voxel-spacing",
        type=parse_target_voxel_size,
        default=DEFAULT_TARGET_VOXEL_SIZE_UM,
        metavar="Z,Y,X",
        help="Target physical voxel size in Z,Y,X order, in µm per voxel",
    )
    parser.add_argument(
        "--normalize-input", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument(
        "--max-training-volumes",
        type=int,
        default=None,
        help="Use at most this many training volumes after the fold split (intended for smoke tests)",
    )
    parser.add_argument(
        "--max-validation-volumes",
        type=int,
        default=None,
        help="Use at most this many validation/test volumes after the fold split (intended for smoke tests)",
    )
    parser.add_argument("--foreground-patch-probability", type=float, default=0.8)
    parser.add_argument(
        "--random-rotation-degrees",
        type=nonnegative_float,
        default=10.0,
        help="Maximum continuous training-patch rotation in the XY plane (degrees)",
    )
    parser.add_argument(
        "--random-rotation-90-probability",
        type=probability,
        default=0.5,
        help="Probability of an exact 0/90/180/270-degree XY rotation",
    )
    return parser


def main():
    args = build_parser().parse_args()
    params = vars(args)
    validate_parent_initialization(args.initial_weights, args.parent_metadata)
    MLFCore.set_general_random_seeds(args.general_seed)
    MLFCore.set_pytorch_random_seeds(args.pytorch_seed)
    mlflow.pytorch.autolog()
    mlflow.start_run()
    MLFCore.log_runtime_environment()
    data = NumorphDataModule(**params)
    data.prepare_data()
    data.setup("fit")
    selected_fold = data.validation_fold + 1
    mlflow.log_param("selected_validation_fold", selected_fold)
    print(
        f"[bold blue]Cross-validation fold: [bold green]{selected_fold}/{args.cross_validation_folds}"
    )
    MLFCore.log_input_data(args.dataset_path)
    model = NumorphSegmentator(**params)
    if args.parent_metadata:
        parent_metadata = Path(args.parent_metadata)
        mlflow.log_artifact(str(parent_metadata), artifact_path="parent-model")
    output = Path(
        "/mlruns" if "MLF_CORE_DOCKER_RUN" in os.environ else "lightning_logs"
    )
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", save_top_k=1, monitor="val_iou_1", mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        # Keep deterministic kernels where PyTorch provides them, but allow
        # operations such as CUDA's 3D cross-entropy backward pass that do not
        # currently have a deterministic implementation.
        deterministic="warn",
        benchmark=False,
        callbacks=[checkpoint, lr_monitor],
        logger=TensorBoardLogger(output),
        log_every_n_steps=args.log_interval,
        check_val_every_n_epoch=args.test_epochs,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")
    checkpoint_state = torch.load(
        checkpoint.best_model_path, map_location="cpu", weights_only=True
    )["state_dict"]
    inference_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint_state.items()
        if key.startswith("model.")
    }
    inference_checkpoint = output / "numorph_unet3d.pt"
    torch.save(inference_state, inference_checkpoint)  # nosec B614
    mlflow.log_artifact(str(inference_checkpoint), artifact_path="model")
    mlflow.end_run()
    print(f"[bold blue]TensorBoard logs: [bold green]{output}")


if __name__ == "__main__":
    main()
