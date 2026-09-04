import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from numorph_nuclei_segmentation.data_loading.data_loader import (
    VolumeDataset,
    VolumePair,
    _download_url,
    cross_validation_split,
    download_dataset,
    extract_dataset_archive,
    read_bia_pairs,
)
from numorph_nuclei_segmentation.numorph_nuclei_segmentation import build_parser
from numorph_nuclei_segmentation.model.unet_3d_models import UNet3D
from numorph_nuclei_segmentation.mlf_core.mlf_core import MLFCore


def write_tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    image = np.arange(60, dtype=np.uint16).reshape(2, 5, 6)
    mask = (image > 10).astype(np.uint8)
    tifffile.imwrite(
        root / "image.ome.tiff",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        root / "mask.ome.tiff",
        mask,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    write_tsv(
        root / "images.tsv",
        [{"Image ID": "volume-1", "File Path": "image.ome.tiff", "split": "train"}],
    )
    write_tsv(
        root / "annotations.tsv",
        [
            {
                "annotation_id": "mask-1",
                "Source Image": "volume-1",
                "File": "mask.ome.tiff",
            }
        ],
    )


def test_bia_bioimageio_metadata_loads_preprocessed_ome_tiffs(tmp_path: Path):
    make_dataset(tmp_path / "bioimageio-wrapper" / "dataset")
    pairs = read_bia_pairs(tmp_path)
    loaded_image, loaded_mask = VolumeDataset(
        pairs, patch_size=(4, 8, 8), normalize=True
    )[0]
    assert loaded_image.shape == (1, 4, 8, 8)
    assert loaded_image.dtype == torch.float32
    assert loaded_image.min() == 0 and loaded_image.max() == 1
    assert loaded_mask.shape == (4, 8, 8)
    assert loaded_mask.dtype == torch.int64
    assert set(loaded_mask.unique().tolist()) <= {0, 1}


def test_zip_archive_is_safely_extracted_and_discovered(tmp_path: Path):
    source = tmp_path / "source" / "nested"
    make_dataset(source)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in source.iterdir():
            zipped.write(path, f"submission/nested/{path.name}")
    extracted = extract_dataset_archive(archive)
    try:
        assert len(read_bia_pairs(extracted.name)) == 1
    finally:
        extracted.cleanup()


def test_public_dataset_is_downloaded_and_existing_file_is_reused(tmp_path: Path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"zip contents")
    destination = tmp_path / "downloads" / "dataset.zip"

    assert download_dataset(source.as_uri(), destination) == destination
    assert destination.read_bytes() == b"zip contents"
    source.write_bytes(b"changed")
    download_dataset(source.as_uri(), destination)
    assert destination.read_bytes() == b"zip contents"
    download_dataset(source.as_uri(), destination, overwrite=True)
    assert destination.read_bytes() == b"changed"


def test_google_drive_share_link_is_converted_to_direct_download():
    direct = _download_url(
        "https://drive.google.com/file/d/example-id/view?usp=drive_link"
    )

    assert (
        direct
        == "https://drive.usercontent.google.com/download?id=example-id&export=download&confirm=t"
    )


def test_bia_filename_references_and_bioimage_columns_are_supported(tmp_path: Path):
    image = np.zeros((4, 8, 8), dtype=np.uint8)
    tifffile.imwrite(
        tmp_path / "raw.ome.tiff",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        tmp_path / "labels.ome.tiff",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    write_tsv(tmp_path / "images.tsv", [{"image": "raw.ome.tiff"}])
    write_tsv(
        tmp_path / "annotations.tsv",
        [{"source image": "raw.ome.tiff", "annotation": "labels.ome.tiff"}],
    )

    [pair] = read_bia_pairs(tmp_path)

    assert pair.image_id == "raw.ome.tiff"
    assert pair.annotation.name == "labels.ome.tiff"


def test_numorph_submission_layout_is_resolved_from_bia_tables(tmp_path: Path):
    package = tmp_path / "NUMORPH_SEM_SEG_DATASET"
    image_dir = package / "data" / "C075" / "images"
    mask_dir = package / "data" / "C075" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    volume = np.zeros((4, 8, 8), dtype=np.uint8)
    tifffile.imwrite(
        image_dir / "raw.ome.tif",
        volume,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        mask_dir / "mask.ome.tif",
        volume,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    bia = package / "bia"
    bia.mkdir()
    write_tsv(
        bia / "images.tsv",
        [
            {
                "Files": "data/C075/images/raw.ome.tif",
                "Resolution group": "C075",
                "Pair ID": "0001",
            }
        ],
    )
    write_tsv(
        bia / "annotations.tsv",
        [
            {
                "Files": "data/C075/masks/mask.ome.tif",
                "source_image": "data/C075/images/raw.ome.tif",
            }
        ],
    )

    [pair] = read_bia_pairs(tmp_path)

    assert pair.image_id == "C075:0001"
    assert pair.group == "C075"
    assert pair.image == image_dir / "raw.ome.tif"
    assert pair.annotation == mask_dir / "mask.ome.tif"


def test_foreground_sampling_keeps_sparse_nucleus_markers(tmp_path: Path):
    image = np.zeros((8, 16, 16), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[0, 0, 0] = 1
    tifffile.imwrite(
        tmp_path / "raw.ome.tif",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        tmp_path / "mask.ome.tif",
        mask,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    pair = VolumePair("sparse", tmp_path / "raw.ome.tif", tmp_path / "mask.ome.tif")
    dataset = VolumeDataset(
        [pair],
        patch_size=(4, 8, 8),
        random_crop=True,
        classes=2,
        foreground_probability=1.0,
        samples_per_volume=2,
    )

    assert len(dataset) == 2
    assert int(dataset[0][1].sum()) == 1


def test_random_rotation_preserves_shape_dtype_and_mask_classes(tmp_path: Path):
    image = np.zeros((8, 16, 16), dtype=np.float32)
    image[:, 4:12, 4:12] = 1
    mask = image.astype(np.uint8)
    tifffile.imwrite(
        tmp_path / "raw.ome.tif",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        tmp_path / "mask.ome.tif",
        mask,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    pair = VolumePair("rotate", tmp_path / "raw.ome.tif", tmp_path / "mask.ome.tif")

    rotated_image, rotated_mask = VolumeDataset(
        [pair], patch_size=(8, 16, 16), random_crop=True, random_rotation_degrees=2.0
    )[0]

    assert rotated_image.shape == (1, 8, 16, 16)
    assert rotated_image.dtype == torch.float32
    assert rotated_mask.shape == (8, 16, 16)
    assert rotated_mask.dtype == torch.int64
    assert set(rotated_mask.unique().tolist()) <= {0, 1}


def test_non_binary_mask_values_are_rejected(tmp_path: Path):
    image = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1, 2, 3] = 2
    tifffile.imwrite(
        tmp_path / "raw.ome.tif",
        image,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    tifffile.imwrite(
        tmp_path / "mask.ome.tif",
        mask,
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 3.2,
            "PhysicalSizeY": 1.0,
            "PhysicalSizeX": 1.0,
        },
    )
    pair = VolumePair("non-binary", tmp_path / "raw.ome.tif", tmp_path / "mask.ome.tif")

    with pytest.raises(ValueError, match=r"binary voxel labels 0 and 1 only.*\[2\]"):
        VolumeDataset([pair])[0]


def test_training_augmentation_defaults_to_requested_patch_and_rotation():
    args = build_parser().parse_args([])

    assert args.patch_size == "32,128,128"
    assert args.random_rotation_degrees == 10.0
    assert args.random_rotation_90_probability == 0.5
    assert args.target_voxel_spacing == (3.0, 1.0, 1.0)
    assert args.loss_function == "dice-ce"
    assert args.cross_validation_folds == 5
    assert args.validation_fold == 0
    assert args.dropout_rate == 0.10


def test_cross_validation_uses_seeded_random_fold_and_all_samples():
    pairs = [
        VolumePair(
            f"sample-{index}",
            Path(f"image-{index}"),
            Path(f"mask-{index}"),
            group=f"group-{index % 2}",
        )
        for index in range(10)
    ]

    train, validation, fold = cross_validation_split(
        pairs, n_folds=5, validation_fold=None, seed=7
    )
    repeated = cross_validation_split(pairs, n_folds=5, validation_fold=None, seed=7)

    assert fold == repeated[2]
    assert [pair.image_id for pair in validation] == [
        pair.image_id for pair in repeated[1]
    ]
    assert len(train) == 8
    assert len(validation) == 2
    assert {pair.image_id for pair in train}.isdisjoint(
        pair.image_id for pair in validation
    )
    assert {pair.image_id for pair in train + validation} == {
        pair.image_id for pair in pairs
    }


def test_cross_validation_accepts_an_explicit_fold_and_validates_fold_count():
    pairs = [
        VolumePair(str(index), Path(str(index)), Path(str(index))) for index in range(5)
    ]

    _, validation, selected = cross_validation_split(
        pairs, n_folds=5, validation_fold=3, seed=0
    )

    assert selected == 3
    assert len(validation) == 1
    with pytest.raises(ValueError, match="cannot exceed"):
        cross_validation_split(pairs, n_folds=6, validation_fold=None, seed=0)
    with pytest.raises(ValueError, match="between 1 and 5"):
        cross_validation_split(pairs, n_folds=5, validation_fold=5, seed=0)


def test_unknown_image_reference_is_rejected(tmp_path: Path):
    write_tsv(
        tmp_path / "images.tsv",
        [{"image_id": "volume-1", "filename": "image.ome.tiff"}],
    )
    write_tsv(
        tmp_path / "annotations.tsv",
        [{"image_id": "missing", "filename": "mask.ome.tiff"}],
    )
    with pytest.raises(ValueError, match="unknown image"):
        read_bia_pairs(tmp_path)


def test_training_model_matches_inference_unet_contract():
    model = UNet3D(in_channels=1, classes=3, dropout=0.25)
    output = model(torch.zeros(1, 1, 4, 8, 8))
    assert output.shape == (1, 3, 4, 8, 8)
    assert {"inc.conv_block_1.conv_1.weight", "outc.conv_1.weight"} <= set(
        model.state_dict()
    )


def test_zip_dataset_itself_is_included_in_provenance(tmp_path: Path):
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"dataset")

    checksums = MLFCore.get_md5_sums(archive)

    assert checksums == [(str(archive), MLFCore.md5(archive))]
