import torch

from numorph_nuclei_segmentation.metrics.metrics import (
    SegmentationConfusionMetric,
    confusion_counts,
    metrics_from_confusion,
)


def test_iou_is_invariant_to_batch_partition_and_merged_rank_counts():
    prediction = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    target = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1])
    complete = confusion_counts(prediction, target, 2)

    for boundaries in ((3,), (2, 5), (1, 3, 6)):
        partials = []
        start = 0
        for end in (*boundaries, prediction.numel()):
            partials.append(
                confusion_counts(prediction[start:end], target[start:end], 2)
            )
            start = end
        merged = torch.stack(partials).sum(0)
        assert torch.equal(merged, complete)
        for actual, expected in zip(
            metrics_from_confusion(merged), metrics_from_confusion(complete)
        ):
            assert torch.equal(actual, expected)


def test_zero_union_class_has_zero_iou():
    counts = confusion_counts(torch.zeros(4), torch.zeros(4), 2)
    iou, mean_iou, accuracy = metrics_from_confusion(counts)
    assert torch.equal(iou, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert mean_iou == 0.5
    assert accuracy == 1.0


def test_metric_accumulates_exact_counts_and_resets():
    metric = SegmentationConfusionMetric(2)
    prediction = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 0, 0])
    expected = confusion_counts(prediction, target, 2)

    metric.update(prediction[:2], target[:2])
    metric.update(prediction[2:], target[2:])
    assert torch.equal(metric.counts, expected)
    for actual, wanted in zip(metric.compute(), metrics_from_confusion(expected)):
        assert torch.equal(actual, wanted)

    metric.reset()
    assert torch.equal(metric.counts, torch.zeros_like(expected))
