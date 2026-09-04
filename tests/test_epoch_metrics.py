import pytest
import torch

from numorph_nuclei_segmentation.model.model import NumorphSegmentator
from numorph_nuclei_segmentation.numorph_nuclei_segmentation import build_parser


def _model():
    return NumorphSegmentator(**vars(build_parser().parse_args([])))


def _update(metrics, prediction, target):
    metrics[0].update(prediction, target)
    metrics[1].update((prediction == 0).int(), (target == 0).int())
    metrics[2].update((prediction == 1).int(), (target == 1).int())
    metrics[3].update(prediction, target)


def _phase_metrics(model, phase):
    return tuple(
        getattr(model, f"{phase}_{name}")
        for name in ("accuracy", "iou_0", "iou_1", "mean_iou")
    )


def test_scalar_metrics_match_known_confusion_matrix():
    prediction = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    target = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1])
    metrics = _phase_metrics(_model(), "train")

    _update(metrics, prediction, target)

    assert [float(metric.compute()) for metric in metrics] == pytest.approx(
        [0.75, 0.6, 0.6, 0.6]
    )


def test_absent_foreground_uses_zero_division_without_non_finite_values():
    target = prediction = torch.zeros(8, dtype=torch.long)
    metrics = _phase_metrics(_model(), "val")

    _update(metrics, prediction, target)
    values = torch.stack([metric.compute() for metric in metrics])

    assert torch.isfinite(values).all()
    # Multiclass macro averaging ignores absent classes in TorchMetrics 1.4.2;
    # the dedicated foreground metric still applies zero_division=0.
    assert values.tolist() == pytest.approx([1.0, 1.0, 0.0, 1.0])


def test_phases_have_independent_metric_objects_and_state():
    model = _model()
    phase_metrics = [
        _phase_metrics(model, phase) for phase in ("train", "val", "test")
    ]
    assert len({id(metric) for metrics in phase_metrics for metric in metrics}) == 12

    _update(
        phase_metrics[0],
        torch.ones(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
    )

    assert all(metric.update_count == 1 for metric in phase_metrics[0])
    assert all(
        metric.update_count == 0
        for metrics in phase_metrics[1:]
        for metric in metrics
    )
