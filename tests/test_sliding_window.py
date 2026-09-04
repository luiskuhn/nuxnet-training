import pytest
import torch

from numorph_nuclei_segmentation.metrics.metrics import confusion_counts
from numorph_nuclei_segmentation.model.model import sliding_window_inference


class PointwisePredictor(torch.nn.Module):
    def forward(self, image):
        return torch.cat((-image, image), dim=1)


@pytest.mark.parametrize("shape", [(8, 12, 12), (3, 5, 6), (7, 11, 13)])
def test_sliding_window_preserves_shape_is_deterministic_and_covers_all_voxels(shape):
    image = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(
        1, 1, *shape
    )
    predictor = PointwisePredictor()
    first = sliding_window_inference(image, predictor, (4, 8, 8), 0.5, 1)
    repeated = sliding_window_inference(image, predictor, (4, 8, 8), 0.5, 3)

    assert first.shape == (1, 2, *shape)
    assert torch.equal(first, repeated)
    assert torch.equal(first, predictor(image))


def test_one_window_is_equivalent_to_direct_inference():
    image = torch.randn(1, 1, 4, 8, 8)
    predictor = PointwisePredictor()
    assert torch.equal(
        sliding_window_inference(image, predictor, (4, 8, 8), 0.75, 2),
        predictor(image),
    )


def test_signal_outside_the_old_center_patch_contributes_to_counts():
    image = torch.full((1, 1, 4, 8, 16), -1.0)
    target = torch.zeros((1, 4, 8, 16), dtype=torch.long)
    image[..., 0] = 1.0
    target[..., 0] = 1
    prediction = sliding_window_inference(
        image, PointwisePredictor(), (4, 8, 8), 0.5, 2
    ).argmax(1)
    counts = confusion_counts(prediction, target, 2)
    assert counts[1, 0] == 32
    assert counts[1, 2] == 0


@pytest.mark.parametrize("overlap", [-0.1, 1.0])
def test_sliding_window_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap"):
        sliding_window_inference(
            torch.zeros(1, 1, 4, 8, 8), PointwisePredictor(), (4, 8, 8), overlap
        )
