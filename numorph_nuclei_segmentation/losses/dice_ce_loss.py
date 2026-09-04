"""Combined cross-entropy and foreground soft-Dice objective."""

import torch
from torch import nn


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None, smooth=1e-5):
        super().__init__()
        weights = (
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32)
        )
        self.register_buffer("class_weights", weights)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        target = target.long()
        cross_entropy = nn.functional.cross_entropy(
            logits, target, weight=self.class_weights
        )
        probability = torch.softmax(logits, dim=1)[:, 1]
        target_fg = (target == 1).to(probability.dtype)
        intersection = (probability * target_fg).sum()
        denominator = probability.sum() + target_fg.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return self.ce_weight * cross_entropy + self.dice_weight * dice_loss
