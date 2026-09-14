"""Ensure NuxNet consumes installed readers instead of another vendored copy."""

from importlib.resources import files
from pathlib import Path

from nidavellir_tools import data_loading
from numorph_nuclei_segmentation.data_loading import data_loader


def test_nuxnet_reexports_installed_dataset_readers():
    for name in (
        "VolumePair", "OMEVolume", "read_bia_pairs", "extract_dataset_archive", "_read_ome"
    ):
        assert getattr(data_loader, name) is getattr(data_loading, name)


def test_package_is_not_vendored_and_resources_are_installed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "nidavellir_tools").exists()
    assert not Path(data_loading.__file__).resolve().is_relative_to(root)
    resources = files("nidavellir_tools")
    assert resources.joinpath("model-card-template.md").is_file()
    assert resources.joinpath("examples/model-package.example.yaml").is_file()
