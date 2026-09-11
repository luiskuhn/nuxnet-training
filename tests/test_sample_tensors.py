import numpy as np
import pytest
import tifffile

from nidavellir_tools.sample_tensors import create_sample_tensor


@pytest.mark.parametrize(
    "shape,axes,stored_axes,stored_shape",
    [
        ((2, 1, 4, 5), "bcyx", "YX", (4, 5)),
        ((2, 3, 4, 5), "bcyx", "CYX", (3, 4, 5)),
        ((2, 1, 3, 4, 5), "bczyx", "ZYX", (3, 4, 5)),
        ((2, 3, 3, 4, 5), "bczyx", "CZYX", (3, 3, 4, 5)),
    ],
)
def test_standard_layout_round_trip(tmp_path, shape, axes, stored_axes, stored_shape):
    array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = tmp_path / "sample.tif"
    record = create_sample_tensor(array, list(axes), path, layout=axes, batch_index=1)
    assert record["tiff_axes"] == stored_axes
    assert tifffile.imread(path).shape == stored_shape
    with tifffile.TiffFile(path) as tif:
        assert tif.series[0].axes == stored_axes


def test_auto_reorders_rdf_axes_and_can_preserve_channel(tmp_path):
    array = np.arange(2 * 4 * 5 * 3, dtype=np.uint16).reshape(2, 4, 5, 3)
    path = tmp_path / "reordered.tif"
    create_sample_tensor(
        array, ["batch", "y", "x", "channel"], path,
        channel_policy="preserve",
    )
    expected = np.transpose(array[0], (2, 0, 1))
    np.testing.assert_array_equal(tifffile.imread(path), expected)


@pytest.mark.parametrize(
    "array,axes,kwargs,message",
    [
        (np.zeros((1, 1, 2)), "bcx", {}, "X and Y"),
        (np.zeros((1, 1, 2, 3)), "bcyx", {"layout": "bczyx"}, "does not match"),
        (np.zeros((1, 1, 2, 3)), "bcyx", {"batch_index": 1}, "outside"),
        (np.zeros((1, 1, 2, 3, 4, 5)), "bctzyx", {}, "unsupported"),
        (np.zeros((1, 2, 3)), "byx", {}, "batch and channel"),
    ],
)
def test_invalid_image_axes_fail_clearly(tmp_path, array, axes, kwargs, message):
    with pytest.raises((ValueError, IndexError), match=message):
        create_sample_tensor(array, list(axes), tmp_path / "bad.tif", **kwargs)
