"""PyTorch Lightning module for 3D U-Net training."""

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics.classification import (
    BinaryJaccardIndex,
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)

from numorph_nuclei_segmentation.losses.dice_ce_loss import DiceCrossEntropyLoss
from numorph_nuclei_segmentation.model.unet_3d_models import UNet3D


class NumorphSegmentator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        test_epochs = kwargs.get("test_epochs")
        if (
            isinstance(test_epochs, bool)
            or not isinstance(test_epochs, int)
            or test_epochs < 1
        ):
            raise ValueError("test_epochs must be a positive integer")
        self.args = kwargs
        self.save_hyperparameters(kwargs)
        self.model = UNet3D(
            in_channels=kwargs["n_channels"],
            classes=kwargs["n_class"],
            dropout=kwargs["dropout_rate"],
        )
        initial_weights = kwargs.get("initial_weights")
        if initial_weights:
            state = torch.load(initial_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        weights = np.asarray(
            [
                float(value)
                for value in kwargs.get("class_weights", "1.0,1.0").split(",")
            ]
        )
        if len(weights) != kwargs["n_class"]:
            raise ValueError("class-weights must contain one value per class")
        # 1,1 is intentionally treated as unweighted CE; non-uniform values
        # remain available only when explicitly requested.
        ce_weights = None if np.allclose(weights, np.ones_like(weights)) else weights
        self.criterion = DiceCrossEntropyLoss(
            kwargs.get("ce_loss_weight", 1.0),
            kwargs.get("dice_loss_weight", 1.0),
            ce_weights,
        )
        if kwargs["n_class"] != 2:
            raise ValueError("segmentation metrics require exactly two classes")
        for phase in ("train", "val", "test"):
            setattr(
                self,
                f"{phase}_accuracy",
                MulticlassAccuracy(num_classes=2, average="micro"),
            )
            setattr(
                self,
                f"{phase}_iou_0",
                BinaryJaccardIndex(threshold=0.5, zero_division=0),
            )
            setattr(
                self,
                f"{phase}_iou_1",
                BinaryJaccardIndex(threshold=0.5, zero_division=0),
            )
            setattr(
                self,
                f"{phase}_mean_iou",
                MulticlassJaccardIndex(
                    num_classes=2, average="macro", zero_division=0
                ),
            )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, phase, *, sliding=False):
        image, target = batch
        logits = (
            sliding_window_inference(
                image,
                self.model,
                self._patch_size(),
                self.args.get("inference_overlap", 0.0),
                self.args.get("test_batch_size", 1),
            )
            if sliding
            else self(image)
        )
        prediction = logits.argmax(dim=1)
        target = target.long()
        loss = self.criterion(logits, target)
        batch_size = image.shape[0]
        self.log(
            f"{phase}_avg_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        accuracy = getattr(self, f"{phase}_accuracy")
        iou_0 = getattr(self, f"{phase}_iou_0")
        iou_1 = getattr(self, f"{phase}_iou_1")
        mean_iou = getattr(self, f"{phase}_mean_iou")
        accuracy.update(prediction, target)
        iou_0.update((prediction == 0).int(), (target == 0).int())
        iou_1.update((prediction == 1).int(), (target == 1).int())
        mean_iou.update(prediction, target)
        for name, metric in (
            ("avg_acc", accuracy),
            ("iou_0", iou_0),
            ("iou_1", iou_1),
            ("mean_iou", mean_iou),
        ):
            self.log(
                f"{phase}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
            )
        return loss

    def _patch_size(self):
        value = self.args["patch_size"]
        return (
            tuple(int(part) for part in value.split(","))
            if isinstance(value, str)
            else tuple(value)
        )

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val", sliding=True)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test", sliding=True)

    def configure_optimizers(self):
        """Configure Adam and observe the plateau only after each validation run.

        ``test_epochs`` is both Lightning's validation interval and the scheduler
        frequency, so plateau patience and cooldown count validation observations
        rather than training epochs.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.args["lr_scheduler_factor"],
            patience=self.args["lr_scheduler_patience"],
            threshold=self.args["lr_scheduler_threshold"],
            threshold_mode="abs",
            cooldown=self.args["lr_scheduler_cooldown"],
            min_lr=self.args["min_lr"],
            eps=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou_1",
                "interval": "epoch",
                "frequency": self.args["test_epochs"],
            },
        }


def _window_starts(length, window, overlap):
    """Return deterministic starts that cover an axis and end at its border."""
    if length <= window:
        return [0]
    stride = max(1, int(window * (1.0 - overlap)))
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def sliding_window_inference(image, predictor, patch_size, overlap=0.0, batch_size=1):
    """Reconstruct raw logits for one or more CZYX volumes from bounded windows.

    Uniform normalized accumulation covers borders exactly. Smaller volumes are
    padded on their high edges and cropped back to their original shape.
    """
    if not 0 <= overlap < 1:
        raise ValueError("inference overlap must be in [0, 1)")
    if batch_size < 1:
        raise ValueError("inference window batch size must be at least 1")
    outputs = []
    for volume in image:
        original = volume.shape[-3:]
        padding = []
        for actual, window in reversed(list(zip(original, patch_size))):
            padding.extend((0, max(0, window - actual)))
        padded = torch.nn.functional.pad(volume, padding)
        shape = padded.shape[-3:]
        locations = [
            (z, y, x)
            for z in _window_starts(shape[0], patch_size[0], overlap)
            for y in _window_starts(shape[1], patch_size[1], overlap)
            for x in _window_starts(shape[2], patch_size[2], overlap)
        ]
        logits_sum = weights = None
        for offset in range(0, len(locations), batch_size):
            group = locations[offset : offset + batch_size]
            windows = torch.stack(
                [
                    padded[
                        :,
                        z : z + patch_size[0],
                        y : y + patch_size[1],
                        x : x + patch_size[2],
                    ]
                    for z, y, x in group
                ]
            )
            logits = predictor(windows)
            if logits_sum is None:
                logits_sum = logits.new_zeros((logits.shape[1], *shape))
                weights = logits.new_zeros(shape)
            for result, (z, y, x) in zip(logits, group):
                region = (
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                logits_sum[(slice(None), *region)] += result
                weights[region] += 1
        outputs.append(
            (logits_sum / weights.unsqueeze(0))[
                :, : original[0], : original[1], : original[2]
            ]
        )
    return torch.stack(outputs)
