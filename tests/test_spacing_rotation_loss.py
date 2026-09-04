import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from numorph_nuclei_segmentation.data_loading.data_loader import (
    _random_rotate,
    resample_volume_pair,
)
from numorph_nuclei_segmentation.losses import DiceCrossEntropyLoss
from numorph_nuclei_segmentation.numorph_nuclei_segmentation import build_parser


def test_trainer_warns_instead_of_failing_for_nondeterministic_cuda_ops():
    entrypoint = (
        Path(__file__).parents[1]
        / "numorph_nuclei_segmentation"
        / "numorph_nuclei_segmentation.py"
    )
    tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
    trainer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Trainer"
    ]

    assert len(trainer_calls) == 1
    deterministic = next(
        keyword.value
        for keyword in trainer_calls[0].keywords
        if keyword.arg == "deterministic"
    )
    assert isinstance(deterministic, ast.Constant)
    assert deterministic.value == "warn"


def _physical_object(spacing):
    shape = tuple(round(25.6 / value) for value in spacing)
    image = np.ones((1, *shape), dtype=np.float32)
    mask = np.ones(shape, dtype=np.uint8)
    return resample_volume_pair(image, mask, spacing, (3.0, 1.0, 1.0))


def test_both_source_resolutions_map_objects_to_the_same_grid():
    high_image, high_mask = _physical_object((2.5, 0.75, 0.75))
    low_image, low_mask = _physical_object((4.0, 1.21, 1.21))
    assert high_image.shape[1:] == high_mask.shape
    assert low_image.shape[1:] == low_mask.shape
    assert all(abs(a - b) <= 1 for a, b in zip(high_mask.shape, low_mask.shape))


def test_resampling_preserves_thin_marker_components():
    image = np.zeros((1, 8, 24, 24), dtype=np.float32)
    mask = np.zeros((8, 24, 24), dtype=np.uint8)
    mask[1, 3, 3] = 1
    mask[6, 18, 18] = 1
    _, result = resample_volume_pair(image, mask, (1.0, 1.0, 1.0), (3.0, 1.0, 1.0))
    assert result.sum() >= 2


def test_resampling_skips_component_scan_when_the_mask_cannot_shrink():
    image = np.ones((1, 4, 8, 8), dtype=np.float32)
    mask = np.ones((4, 8, 8), dtype=np.uint8)

    with patch(
        "numorph_nuclei_segmentation.data_loading.data_loader._component_centroids",
        side_effect=AssertionError("component scan should not run"),
    ):
        _, result = resample_volume_pair(
            image, mask, (3.0, 1.0, 1.0), (3.0, 1.0, 1.0)
        )

    assert np.array_equal(result, mask)


def test_exact_rotation_is_xy_only_and_binary(monkeypatch):
    image = np.arange(2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    mask = (image[0] % 3 == 0).astype(np.uint8)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(1))
    rotated_image, rotated_mask = _random_rotate(image, mask, 0, 1.0)
    assert torch.equal(rotated_image, torch.rot90(torch.from_numpy(image), 1, (-2, -1)))
    assert torch.equal(
        rotated_mask, torch.rot90(torch.from_numpy(mask).long(), 1, (-2, -1))
    )
    assert torch.equal(
        rotated_image[:, 0], torch.rot90(torch.from_numpy(image[:, 0]), 1, (-2, -1))
    )


def test_exact_rotation_keeps_rectangular_patch_shape(monkeypatch):
    image = np.zeros((1, 2, 4, 6), dtype=np.float32)
    mask = np.zeros((2, 4, 6), dtype=np.uint8)
    image[:, :, 1:3, 2:4] = 1
    mask[:, 1:3, 2:4] = 1
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(1))

    rotated_image, rotated_mask = _random_rotate(image, mask, 0, 1.0)

    assert rotated_image.shape == image.shape
    assert rotated_mask.shape == mask.shape
    assert torch.equal(rotated_image[0] > 0, rotated_mask.bool())


def test_continuous_xy_rotation_keeps_z_and_binary_mask():
    image = np.zeros((1, 3, 12, 12), dtype=np.float32)
    image[:, 1, 3:9, 5:7] = 1
    mask = image[0].astype(np.uint8)
    rotated_image, rotated_mask = _random_rotate(image, mask, 10, 0)
    assert rotated_image.shape == image.shape
    assert rotated_mask.shape == mask.shape
    assert set(rotated_mask.unique().tolist()) <= {0, 1}
    assert rotated_mask[0].sum() == rotated_mask[2].sum() == 0


def test_dice_ce_is_finite_with_and_without_foreground_and_backpropagates():
    criterion = DiceCrossEntropyLoss()
    for foreground in (False, True):
        logits = torch.randn(2, 2, 3, 4, 4, requires_grad=True)
        target = torch.zeros(2, 3, 4, 4, dtype=torch.long)
        if foreground:
            target[:, 1, 2, 2] = 1
        loss = criterion(logits, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()


def test_dice_ce_cli_values_have_expected_defaults():
    args = build_parser().parse_args([])
    assert args.target_voxel_spacing == (3.0, 1.0, 1.0)
    assert args.random_rotation_degrees == 10.0
    assert args.random_rotation_90_probability == 0.5
    assert vars(args)["ce_loss_weight"] == vars(args)["dice_loss_weight"] == 1.0
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--loss-function", "focal"])


def test_target_spacing_cli_parses_valid_values_and_rejects_invalid_values():
    parser = build_parser()
    args = parser.parse_args(["--target-voxel-spacing", "4.0, 1.2, 1.2"])
    assert args.target_voxel_spacing == (4.0, 1.2, 1.2)
    with pytest.raises(SystemExit):
        parser.parse_args(["--target-voxel-spacing", "3.0,1.0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--target-voxel-spacing", "3.0,0,1.0"])


def test_new_cli_hyperparameters_are_serializable_and_range_checked():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--target-voxel-spacing",
            "4,1.2,1.2",
            "--random-rotation-degrees",
            "7.5",
            "--random-rotation-90-probability",
            "0.25",
            "--ce-loss-weight",
            "0.5",
            "--dice-loss-weight",
            "2",
        ]
    )
    serialized = json.loads(json.dumps(vars(args)))
    assert serialized["target_voxel_spacing"] == [4.0, 1.2, 1.2]
    assert serialized["random_rotation_degrees"] == 7.5
    assert serialized["random_rotation_90_probability"] == 0.25
    with pytest.raises(SystemExit):
        parser.parse_args(["--random-rotation-90-probability", "1.1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dice-loss-weight", "-1"])
