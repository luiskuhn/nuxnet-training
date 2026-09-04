"""PyTorch Lightning module for 3D U-Net training."""

import numpy as np
import pytorch_lightning as pl
import torch

from numorph_nuclei_segmentation.losses.focal_loss import FocalLoss
from numorph_nuclei_segmentation.losses.dice_ce_loss import DiceCrossEntropyLoss
from numorph_nuclei_segmentation.metrics.metrics import accuracy, iou_fnc
from numorph_nuclei_segmentation.model.unet_3d_models import UNet3D


class NumorphSegmentator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.save_hyperparameters(kwargs)
        self.model = UNet3D(
            in_channels=kwargs["n_channels"],
            classes=kwargs["n_class"],
            dropout=kwargs["dropout_rate"],
        )
        weights = np.asarray(
            [
                float(value)
                for value in kwargs.get("class_weights", "1.0,1.0").split(",")
            ]
        )
        if len(weights) != kwargs["n_class"]:
            raise ValueError("class-weights must contain one value per class")
        if kwargs.get("loss_function", "dice-ce") == "focal":
            self.criterion = FocalLoss(
                apply_nonlin=torch.nn.Softmax(dim=1),
                alpha=weights,
                gamma=2,
            )
        else:
            # 1,1 is intentionally treated as unweighted CE; non-uniform values
            # remain available only when explicitly requested.
            ce_weights = (
                None if np.allclose(weights, np.ones_like(weights)) else weights
            )
            self.criterion = DiceCrossEntropyLoss(
                kwargs.get("ce_loss_weight", 1.0),
                kwargs.get("dice_loss_weight", 1.0),
                ce_weights,
            )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, prefix):
        image, target = batch
        logits = self(image)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).float()
        loss = self.criterion(logits, target.long())
        batch_size = image.shape[0]
        self.log(
            f"{prefix}_avg_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}_avg_acc",
            float(accuracy(prediction, target)),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        iou, counts = iou_fnc(prediction, target, self.args["n_class"])
        valid_scores = []
        for class_id in range(self.args["n_class"]):
            score = (
                float(iou[class_id] / (counts[class_id] + 1e-10))
                if counts[class_id]
                else 0.0
            )
            if counts[class_id]:
                valid_scores.append(score)
            self.log(
                f"{prefix}_iou_{class_id}",
                score,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            f"{prefix}_mean_iou",
            sum(valid_scores) / max(1, len(valid_scores)),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_avg_loss"},
        }
