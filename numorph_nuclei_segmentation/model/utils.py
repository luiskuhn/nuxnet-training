"""Utilities for model inference."""

from torch import nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable only 3D dropout layers while keeping the rest of ``model`` in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout3d):
            module.train()
