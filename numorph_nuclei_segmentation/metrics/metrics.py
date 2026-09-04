"""Tensor-only confusion-count metrics for segmentation epochs."""

import torch


def confusion_counts(prediction, target, n_classes):
    """Return per-class ``(TP, FP, FN, TN)`` counts as integer tensors.

    Counts are additive, so callers can sum batches or DDP rank-local partials
    before computing metrics.  A class whose union is zero has IoU zero; mean
    IoU consequently always averages the configured set of classes.
    """
    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()
    counts = torch.zeros((n_classes, 4), dtype=torch.long, device=prediction.device)
    for class_id in range(n_classes):
        predicted = prediction == class_id
        actual = target == class_id
        counts[class_id, 0] = (predicted & actual).sum()
        counts[class_id, 1] = (predicted & ~actual).sum()
        counts[class_id, 2] = (~predicted & actual).sum()
        counts[class_id, 3] = (~predicted & ~actual).sum()
    return counts


def metrics_from_confusion(counts):
    """Compute per-class IoU, mean IoU, and global voxel accuracy."""
    counts = counts.to(torch.float64)
    union = counts[:, :3].sum(dim=1)
    iou = torch.where(union > 0, counts[:, 0] / union, torch.zeros_like(union))
    total = counts[0].sum()
    accuracy = counts[:, 0].sum() / total if total > 0 else total.new_zeros(())
    return iou, iou.mean(), accuracy


# Compatibility wrappers for downstream imports. Epoch aggregation should use
# the tensor functions above rather than these scalar convenience functions.
def accuracy(x, y):
    return (x == y).to(torch.float32).mean().item()


def iou_fnc(pred, target, n_classes=12):
    counts = confusion_counts(pred, target, n_classes)
    iou, _, _ = metrics_from_confusion(counts)
    unions = counts[:, :3].sum(1).ne(0).cpu().numpy()
    return iou.cpu().numpy(), unions
