"""Isolated two-rank Lightning job used by the distributed metric test."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from numorph_nuclei_segmentation.metrics.metrics import confusion_counts
from numorph_nuclei_segmentation.model.model import NumorphSegmentator

PREDICTIONS = torch.tensor(
    [
        [0, 0, 1, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
    ]
)
TARGETS = torch.tensor(
    [
        [0, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 0, 0],
    ]
)


class FixedSegmentationDataset(Dataset):
    def __len__(self):
        return len(PREDICTIONS)

    def __getitem__(self, index):
        image = PREDICTIONS[index].float().reshape(1, 2, 2, 2)
        target = TARGETS[index].reshape(2, 2, 2)
        return image, target


class FixedPredictionNet(torch.nn.Module):
    """A trainable network whose predictions remain fixed for exact assertions."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image):
        foreground = image[:, 0]
        logits = torch.stack((1.0 - foreground, foreground), dim=1) * 8.0
        return logits + self.anchor * 0.0


class InstrumentedSegmentator(NumorphSegmentator):
    def __init__(self):
        super().__init__(
            test_epochs=1,
            n_channels=1,
            n_class=2,
            dropout_rate=0.0,
            class_weights="1.0,1.0",
            ce_loss_weight=1.0,
            dice_loss_weight=1.0,
            patch_size=(2, 2, 2),
            inference_overlap=0.0,
            test_batch_size=1,
            lr=0.0,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=1,
            lr_scheduler_threshold=1e-5,
            lr_scheduler_cooldown=0,
            min_lr=0.0,
        )
        self.model = FixedPredictionNet()
        self.events = {"train_start": [], "train_end": [], "validation": [], "test": []}
        self.recorded_metrics = {}

    def log(self, name, value, *args, **kwargs):
        super().log(name, value, *args, **kwargs)
        if name in {"val_iou_1", "test_iou_1"}:
            self.recorded_metrics[name] = float(value.compute())

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        assert not self.train_confusion.counts.any()
        self.events["train_start"].append(self.current_epoch)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        assert not self.train_confusion.counts.any()
        self.events["train_end"].append(self.current_epoch)

    def on_validation_epoch_end(self):
        sanity = self.trainer.sanity_checking
        super().on_validation_epoch_end()
        assert not self.val_confusion.counts.any()
        self.events["validation"].append(
            {"epoch": self.current_epoch, "sanity": sanity}
        )

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        assert not self.test_confusion.counts.any()
        self.events["test"].append(self.current_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = FixedSegmentationDataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        monitor="val_iou_1",
        mode="max",
        save_top_k=1,
    )
    model = InstrumentedSegmentator()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        max_epochs=2,
        num_sanity_val_steps=1,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader, verbose=False)

    expected = confusion_counts(PREDICTIONS, TARGETS, 2)
    scheduler = trainer.lr_scheduler_configs[0]
    report = {
        "rank": trainer.global_rank,
        "events": model.events,
        "counts": expected.tolist(),
        "val_iou_1": model.recorded_metrics["val_iou_1"],
        "test_iou_1": model.recorded_metrics["test_iou_1"],
        "checkpoint_exists": bool(
            checkpoint.best_model_path and Path(checkpoint.best_model_path).is_file()
        ),
        "scheduler_monitor": scheduler.monitor,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{trainer.global_rank}.json").write_text(json.dumps(report))


if __name__ == "__main__":
    # Lightning re-executes this module for the second DDP rank.
    main()
