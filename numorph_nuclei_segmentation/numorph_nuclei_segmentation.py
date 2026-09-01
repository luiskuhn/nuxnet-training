"""NuxNet training entry point."""

import argparse
import os
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print

from numorph_nuclei_segmentation.data_loading.data_loader import NUMORPH_DATASET_URL, NumorphDataModule
from numorph_nuclei_segmentation.mlf_core.mlf_core import MLFCore
from numorph_nuclei_segmentation.model.model import NumorphSegmentator


def build_parser():
    parser = argparse.ArgumentParser(description="Train NuxNet on BIA-described OME-TIFF volumes")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--devices", default="auto", help="Lightning device count or 'auto'")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--general-seed", type=int, default=0)
    parser.add_argument("--pytorch-seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--training-batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--class-weights", default="0.2,1.0")
    parser.add_argument("--cross-validation-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument(
        "--validation-fold",
        type=int,
        default=0,
        help="One-based fold used for validation and test metrics; 0 selects a fold from --general-seed",
    )
    parser.add_argument("--test-epochs", type=int, default=10)
    parser.add_argument("--dataset-path", default="/data")
    parser.add_argument("--download-dataset", action="store_true", help="Download the dataset to --dataset-path if needed")
    parser.add_argument("--overwrite-dataset", action="store_true", help="Replace an existing downloaded dataset")
    parser.add_argument("--dataset-url", default=NUMORPH_DATASET_URL, help="Public HTTP(S) or Google Drive dataset URL")
    parser.add_argument("--n-channels", type=int, default=1)
    parser.add_argument("--n-class", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.25)
    parser.add_argument("--patch-size", default="32,128,128", help="Training patch dimensions Z,Y,X; each must be divisible by 4")
    parser.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--foreground-patch-probability", type=float, default=0.8)
    parser.add_argument(
        "--random-rotation-degrees",
        type=float,
        default=2.0,
        help="Maximum training-patch rotation in degrees about a random 3-D axis",
    )
    return parser


def main():
    args = build_parser().parse_args()
    params = vars(args)
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
    print(f"[bold blue]Cross-validation fold: [bold green]{selected_fold}/{args.cross_validation_folds}")
    MLFCore.log_input_data(args.dataset_path)
    model = NumorphSegmentator(**params)
    output = Path("/mlruns" if "MLF_CORE_DOCKER_RUN" in os.environ else "lightning_logs")
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", save_top_k=1, monitor="val_avg_loss", mode="min")
    devices = int(args.devices) if args.devices.isdigit() else args.devices
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator=args.accelerator, devices=devices, strategy=args.strategy,
        deterministic=True, benchmark=False, callbacks=[checkpoint], logger=TensorBoardLogger(output),
        log_every_n_steps=args.log_interval, check_val_every_n_epoch=args.test_epochs,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")
    checkpoint_state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)["state_dict"]
    inference_state = {key.removeprefix("model."): value for key, value in checkpoint_state.items() if key.startswith("model.")}
    inference_checkpoint = output / "numorph_unet3d.pt"
    torch.save(inference_state, inference_checkpoint)  # nosec B614
    mlflow.log_artifact(str(inference_checkpoint), artifact_path="model")
    mlflow.end_run()
    print(f"[bold blue]TensorBoard logs: [bold green]{output}")


if __name__ == "__main__":
    main()
