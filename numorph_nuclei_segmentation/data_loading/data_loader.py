"""Load BIA/BioImage.IO OME-TIFF segmentation datasets."""

from __future__ import annotations

import csv
import os
import random
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_FILE_COLUMNS = (
    "filename",
    "file_name",
    "file_path",
    "filepath",
    "path",
    "uri",
    "file",
    "files",
)
_IMAGE_FILE_COLUMNS = (*_FILE_COLUMNS, "image")
_ANNOTATION_FILE_COLUMNS = (*_FILE_COLUMNS, "annotation", "mask", "label")
_IMAGE_ID_COLUMNS = ("image_id", "image_uuid", "id", "name")
_SOURCE_COLUMNS = (
    "image_id",
    "source_image_id",
    "source_image_uuid",
    "source_image",
    "image",
    "source",
)

NUMORPH_DATASET_URL = "https://drive.google.com/file/d/1nwLPXoWEsBb3wLNwXHY3L23UiO-5IIyn/view?usp=drive_link"


def cross_validation_split(
    pairs: Iterable[VolumePair], n_folds: int, validation_fold: int | None, seed: int
) -> tuple[list[VolumePair], list[VolumePair], int]:
    """Return a seeded, group-stratified training/validation fold split.

    ``validation_fold`` is zero based internally. When it is ``None``, the
    seed selects a fold reproducibly. Samples in each metadata group are
    shuffled independently and distributed round-robin across folds.
    """
    pairs = list(pairs)
    if n_folds < 2:
        raise ValueError("cross-validation-folds must be at least 2")
    if n_folds > len(pairs):
        raise ValueError(
            f"cross-validation-folds ({n_folds}) cannot exceed the number of samples ({len(pairs)})"
        )
    if validation_fold is None:
        validation_fold = random.Random(seed).randrange(n_folds)  # nosec B311
    if not 0 <= validation_fold < n_folds:
        raise ValueError(f"validation-fold must be between 1 and {n_folds}")

    assignments: dict[str, int] = {}
    rng = random.Random(seed)  # nosec B311
    groups = sorted({pair.group or "all" for pair in pairs})
    offset = 0
    for group in groups:
        members = [pair for pair in pairs if (pair.group or "all") == group]
        rng.shuffle(members)
        for index, pair in enumerate(members):
            assignments[pair.image_id] = (offset + index) % n_folds
        offset = (offset + len(members)) % n_folds

    validation = [
        pair for pair in pairs if assignments[pair.image_id] == validation_fold
    ]
    training = [pair for pair in pairs if assignments[pair.image_id] != validation_fold]
    return training, validation, validation_fold


def _download_url(url: str) -> str:
    """Convert public Google Drive share links to their file-download form."""
    parsed = urlparse(url)
    if parsed.hostname not in {"drive.google.com", "www.drive.google.com"}:
        return url
    parts = [part for part in parsed.path.split("/") if part]
    file_id = (
        parts[parts.index("d") + 1]
        if "d" in parts and parts.index("d") + 1 < len(parts)
        else None
    )
    if file_id is None:
        file_id = parse_qs(parsed.query).get("id", [None])[0]
    if not file_id:
        raise ValueError(f"Could not find a Google Drive file ID in URL: {url}")
    query = urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return urlunparse(
        ("https", "drive.usercontent.google.com", "/download", "", query, "")
    )


def download_dataset(
    url: str, destination: str | Path, *, overwrite: bool = False
) -> Path:
    """Download a public dataset URL atomically, including Google Drive links."""
    destination = Path(destination).expanduser()
    if destination.exists() and not overwrite:
        if not destination.is_file():
            raise IsADirectoryError(
                f"Dataset download destination is a directory: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(_download_url(url), headers={"User-Agent": "nuxnet-training/1.0"})
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        with urlopen(request) as response, temporary.open("wb") as output:  # nosec B310
            if response.headers.get_content_type() == "text/html":
                raise RuntimeError(
                    "Dataset URL returned HTML instead of a file; check that the link is public"
                )
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError("Dataset download was empty")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True)
class VolumePair:
    image_id: str
    image: Path
    annotation: Path
    split: str | None = None
    group: str | None = None


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        return [
            {
                _normalise_key(str(key)): (value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]


def _column(row: dict[str, str], choices: Iterable[str], table: str) -> str:
    for choice in choices:
        if row.get(choice):
            return row[choice]
    raise ValueError(
        f"{table} must contain one of these populated columns: {', '.join(choices)}"
    )


def _metadata_root(root: Path) -> Path:
    candidates = sorted(root.rglob("images.tsv"))
    matches = [
        path.parent
        for path in candidates
        if (path.parent / "annotations.tsv").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one directory containing images.tsv and annotations.tsv below {root}; found {len(matches)}"
        )
    return matches[0]


def _resolve_file(metadata_root: Path, filename: str) -> Path:
    """Resolve paths relative to the TSV folder or its submission-package root."""
    path = Path(filename)
    if path.is_absolute():
        return path
    local = metadata_root / path
    package_relative = metadata_root.parent / path
    return (
        local if local.exists() or not package_relative.exists() else package_relative
    )


def extract_dataset_archive(archive: str | Path) -> tempfile.TemporaryDirectory:
    """Safely extract a downloaded BIA/BioImage.IO ZIP for this process."""
    temporary = tempfile.TemporaryDirectory(prefix="nuxnet_dataset_")
    destination = Path(temporary.name).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                temporary.cleanup()
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        zipped.extractall(destination)
    return temporary


def read_bia_pairs(root: str | Path) -> list[VolumePair]:
    """Join BIA ``images.tsv`` and ``annotations.tsv`` records by source image ID."""
    metadata_root = _metadata_root(Path(root))
    images = _read_tsv(metadata_root / "images.tsv")
    annotations = _read_tsv(metadata_root / "annotations.tsv")
    image_index: dict[str, tuple[str, Path, str | None, str | None]] = {}
    for row in images:
        filename = _column(row, _IMAGE_FILE_COLUMNS, "images.tsv")
        group = row.get("resolution_group") or row.get("subset") or None
        pair_id = row.get("pair_id")
        image_id = next(
            (row[column] for column in _IMAGE_ID_COLUMNS if row.get(column)), None
        )
        image_id = image_id or (f"{group}:{pair_id}" if group and pair_id else filename)
        if image_id in image_index:
            raise ValueError(f"Duplicate image identifier in images.tsv: {image_id}")
        image = _resolve_file(metadata_root, filename)
        # BIA annotation tables in the wild reference either the stable image
        # identifier, the relative file path, or only the source basename.
        for reference in (image_id, filename, Path(filename).name):
            previous = image_index.get(reference)
            if previous and previous[0] != image_id:
                raise ValueError(
                    f"Ambiguous image reference in images.tsv: {reference}"
                )
            image_index[reference] = (image_id, image, row.get("split") or None, group)

    pairs: list[VolumePair] = []
    seen: set[str] = set()
    for row in annotations:
        source = _column(row, _SOURCE_COLUMNS, "annotations.tsv")
        filename = _column(row, _ANNOTATION_FILE_COLUMNS, "annotations.tsv")
        source_record = image_index.get(source) or image_index.get(Path(source).name)
        if source_record is None:
            raise ValueError(
                f"Annotation references unknown image identifier or path: {source}"
            )
        image_id, image, split, group = source_record
        if image_id in seen:
            raise ValueError(
                f"More than one segmentation annotation for image: {image_id}"
            )
        annotation = _resolve_file(metadata_root, filename)
        for kind, candidate in (("image", image), ("annotation", annotation)):
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"{kind.title()} file for {image_id} not found: {candidate}"
                )
            if not candidate.name.lower().endswith((".ome.tif", ".ome.tiff")):
                raise ValueError(f"{kind.title()} file must be OME-TIFF: {candidate}")
        pairs.append(VolumePair(image_id, image, annotation, split, group))
        seen.add(image_id)
    if not pairs:
        raise ValueError("No image/annotation pairs found in the BIA metadata tables")
    return pairs


_UNIT_TO_UM = {
    "µm": 1.0,
    "um": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
}


def _physical_spacing(ome_xml: str | None, path: Path) -> tuple[float, float, float]:
    """Read OME physical sizes and return micrometres/voxel in Z,Y,X order."""
    try:
        pixels = next(
            element
            for element in ET.fromstring(ome_xml or "").iter()
            if element.tag.endswith("Pixels")
        )
        xyz = []
        for axis in "ZYX":
            value = float(pixels.attrib[f"PhysicalSize{axis}"])
            unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm").lower()
            factor = _UNIT_TO_UM[unit]
            if not np.isfinite(value) or value <= 0:
                raise ValueError
            xyz.append(value * factor)
        return tuple(xyz)
    except (ET.ParseError, StopIteration, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Valid PhysicalSizeX, PhysicalSizeY and PhysicalSizeZ OME metadata is required for {path}"
        ) from error


def _read_ome(path: Path, *, mask: bool, return_spacing: bool = False):
    with tifffile.TiffFile(path) as tif:
        if not tif.is_ome:
            raise ValueError(f"Expected OME-TIFF metadata in {path}")
        array = tif.series[0].asarray()
        axes = tif.series[0].axes.upper()
        spacing = _physical_spacing(tif.ome_metadata, path)
    for axis in "ST":
        if axis in axes:
            index = axes.index(axis)
            if array.shape[index] != 1:
                raise ValueError(
                    f"{path} has non-singleton {axis}; export each scene/timepoint as a separate record"
                )
            array = np.take(array, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]
    if mask and "C" in axes:
        index = axes.index("C")
        if array.shape[index] != 1:
            raise ValueError(f"Segmentation mask must have one channel: {path}")
        array = np.take(array, 0, axis=index)
        axes = axes[:index] + axes[index + 1 :]
    wanted = "ZYX" if mask else "CZYX"
    if not mask and "C" not in axes:
        array, axes = np.expand_dims(array, 0), "C" + axes
    if "Z" not in axes:
        index = axes.index("Y")
        array, axes = np.expand_dims(array, index), axes[:index] + "Z" + axes[index:]
    if set(axes) != set(wanted) or len(axes) != len(wanted):
        raise ValueError(f"Unsupported axes {axes!r} in {path}; expected {wanted}")
    array = np.transpose(array, tuple(axes.index(axis) for axis in wanted))
    return (array, spacing) if return_spacing else array


def _gaussian_blur_for_downsampling(
    image: torch.Tensor, scale: tuple[float, float, float]
) -> torch.Tensor:
    """Apply a separable Gaussian low-pass filter on axes that will shrink."""
    for axis, ratio in enumerate(scale):
        if ratio >= 1:
            continue
        sigma = max(0.5, 0.5 * (1.0 / ratio - 1.0))
        radius = max(1, int(np.ceil(3 * sigma)))
        coordinates = torch.arange(
            -radius, radius + 1, dtype=image.dtype, device=image.device
        )
        kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = radius
        channels = image.shape[1]
        image = F.conv3d(
            image,
            kernel.view(shape).repeat(channels, 1, 1, 1, 1),
            padding=padding,
            groups=channels,
        )
    return image


def _component_centroids(mask: np.ndarray) -> list[np.ndarray]:
    """Return 26-connected foreground component centroids without an extra dependency."""
    remaining = set(map(tuple, np.argwhere(mask > 0)))
    centroids = []
    neighbours = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ]
    while remaining:
        stack, component = [remaining.pop()], []
        while stack:
            voxel = stack.pop()
            component.append(voxel)
            for delta in neighbours:
                candidate = tuple(voxel[i] + delta[i] for i in range(3))
                if candidate in remaining:
                    remaining.remove(candidate)
                    stack.append(candidate)
        centroids.append(np.asarray(component, dtype=np.float64).mean(axis=0))
    return centroids


def resample_volume_pair(
    image: np.ndarray, label: np.ndarray, source_spacing, target_spacing
):
    """Resample a CZYX/ZYX pair and preserve each in-bounds marker component."""
    source = np.asarray(source_spacing, dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    if (
        source.shape != (3,)
        or target.shape != (3,)
        or np.any(source <= 0)
        or np.any(target <= 0)
    ):
        raise ValueError(
            "source and target spacing must be three positive values in Z,Y,X order"
        )
    scale = source / target
    output_shape = tuple(
        max(1, int(round(size * factor))) for size, factor in zip(label.shape, scale)
    )
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None]
    image_tensor = _gaussian_blur_for_downsampling(image_tensor, tuple(scale))
    resized_image = F.interpolate(
        image_tensor, size=output_shape, mode="trilinear", align_corners=False
    )[0]
    resized_label = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(label)).float()[None, None],
        size=output_shape,
        mode="nearest-exact",
    )[0, 0].to(torch.int64)
    # Voxel-centre coordinates map as (p + .5) * source / target - .5.
    for centroid in _component_centroids(label):
        transformed = np.rint((centroid + 0.5) * scale - 0.5).astype(int)
        if np.all(transformed >= 0) and np.all(transformed < np.asarray(output_shape)):
            resized_label[tuple(transformed)] = 1
    return resized_image.numpy(), resized_label.numpy()


def _crop_or_pad(
    image: np.ndarray,
    label: np.ndarray,
    patch: tuple[int, int, int],
    random_crop: bool,
    foreground_probability: float,
):
    padding = [(0, 0)] + [
        (0, max(0, size - actual)) for actual, size in zip(label.shape, patch)
    ]
    image = np.pad(image, padding, mode="constant")
    label = np.pad(label, padding[1:], mode="constant")
    foreground = np.argwhere(label > 0)
    foreground_center = None
    if random_crop and len(foreground) and np.random.random() < foreground_probability:
        foreground_center = foreground[np.random.randint(len(foreground))]
    starts = []
    for axis, (actual, size) in enumerate(zip(label.shape, patch)):
        maximum = actual - size
        if foreground_center is not None:
            start = int(np.clip(foreground_center[axis] - size // 2, 0, maximum))
        elif random_crop and maximum:
            start = int(np.random.randint(maximum + 1))
        else:
            start = maximum // 2
        starts.append(start)
    spatial = tuple(slice(start, start + size) for start, size in zip(starts, patch))
    return image[(slice(None), *spatial)], label[spatial]


def _random_rotate(
    image: np.ndarray,
    label: np.ndarray,
    max_degrees: float,
    rotation_90_probability: float = 0.0,
):
    """Rotate an image/mask pair only in XY (around physical Z).

    The same sampling grid is used for both arrays. Intensities use trilinear
    interpolation, while class IDs use nearest-neighbour interpolation so that
    augmentation cannot create new mask classes.
    """
    image_tensor = torch.from_numpy(np.ascontiguousarray(image))
    label_tensor = torch.from_numpy(np.ascontiguousarray(label)).to(torch.int64)
    original_yx = label_tensor.shape[-2:]
    if torch.rand(()) < rotation_90_probability:
        turns = int(torch.randint(0, 4, ()).item())
        image_tensor = torch.rot90(image_tensor, turns, dims=(-2, -1))
        label_tensor = torch.rot90(label_tensor, turns, dims=(-2, -1))
        # Odd quarter turns swap Y/X. Restore a fixed patch shape using only
        # exact center cropping/padding, keeping image and mask aligned.
        if image_tensor.shape[-2:] != original_yx:
            target_y, target_x = original_yx
            current_y, current_x = image_tensor.shape[-2:]
            start_y = max(0, (current_y - target_y) // 2)
            start_x = max(0, (current_x - target_x) // 2)
            image_tensor = image_tensor[
                ..., start_y : start_y + target_y, start_x : start_x + target_x
            ]
            label_tensor = label_tensor[
                ..., start_y : start_y + target_y, start_x : start_x + target_x
            ]
            pad_y, pad_x = (
                target_y - image_tensor.shape[-2],
                target_x - image_tensor.shape[-1],
            )
            padding = (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2)
            image_tensor = F.pad(image_tensor, padding)
            label_tensor = F.pad(label_tensor, padding)
    if max_degrees == 0:
        return image_tensor.contiguous(), label_tensor.contiguous()
    angle = torch.empty((), dtype=torch.float64).uniform_(-max_degrees, max_degrees)
    angle = torch.deg2rad(angle)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    dimensions = torch.tensor(label_tensor.shape[::-1], dtype=torch.float64)
    rotation = rotation * dimensions[None, :] / dimensions[:, None]
    theta = torch.cat((rotation, torch.zeros((3, 1), dtype=rotation.dtype)), dim=1).to(
        torch.float32
    )[None]
    image_tensor = image_tensor[None]
    label_tensor = label_tensor.to(torch.float32)[None, None]
    grid = F.affine_grid(theta, image_tensor.shape, align_corners=False)
    image_tensor = F.grid_sample(
        image_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    label_tensor = F.grid_sample(
        label_tensor, grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    return image_tensor[0], label_tensor[0, 0].to(torch.int64)


class VolumeDataset(Dataset):
    """Lazily read and preprocess image/mask patches for the inference U-Net."""

    def __init__(
        self,
        pairs: Iterable[VolumePair],
        patch_size=None,
        random_crop=False,
        normalize=True,
        classes=None,
        samples_per_volume=1,
        foreground_probability=0.0,
        random_rotation_degrees=0.0,
        random_rotation_90_probability=0.0,
        target_spacing=(3.0, 1.0, 1.0),
    ):
        self.pairs = list(pairs)
        self.patch_size = tuple(patch_size) if patch_size else None
        self.random_crop = random_crop
        self.normalize = normalize
        self.classes = classes
        self.samples_per_volume = samples_per_volume
        self.foreground_probability = foreground_probability
        self.random_rotation_degrees = float(random_rotation_degrees)
        self.random_rotation_90_probability = float(random_rotation_90_probability)
        self.target_spacing = tuple(target_spacing)
        self._volume_cache = {}

    def __len__(self):
        return len(self.pairs) * self.samples_per_volume

    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        if pair.image_id not in self._volume_cache:
            image, image_spacing = _read_ome(
                pair.image, mask=False, return_spacing=True
            )
            label, mask_spacing = _read_ome(
                pair.annotation, mask=True, return_spacing=True
            )
            if not np.allclose(image_spacing, mask_spacing):
                raise ValueError(
                    f"Image and mask physical spacing differ for {pair.image_id}: {image_spacing} != {mask_spacing}"
                )
            if image.shape[1:] != label.shape:
                raise ValueError(
                    f"Image and mask shapes differ for {pair.image_id}: {image.shape[1:]} != {label.shape}"
                )
            if not np.issubdtype(label.dtype, np.integer):
                if not np.array_equal(label, label.astype(np.int64)):
                    raise ValueError(
                        f"Mask contains non-integer class labels: {pair.annotation}"
                    )
            invalid_labels = np.setdiff1d(np.unique(label), (0, 1))
            if invalid_labels.size:
                raise ValueError(
                    f"Mask for {pair.image_id} must contain binary voxel labels 0 and 1 only; "
                    f"found {invalid_labels.tolist()}"
                )
            image, label = resample_volume_pair(
                image.astype(np.float32, copy=False),
                label,
                image_spacing,
                self.target_spacing,
            )
            self._volume_cache[pair.image_id] = image, label
        image, label = self._volume_cache[pair.image_id]
        if not np.issubdtype(label.dtype, np.integer):
            if not np.array_equal(label, label.astype(np.int64)):
                raise ValueError(
                    f"Mask contains non-integer class labels: {pair.annotation}"
                )
        label = label.astype(np.int64, copy=False)
        invalid_labels = np.setdiff1d(np.unique(label), (0, 1))
        if invalid_labels.size:
            raise ValueError(
                f"Mask for {pair.image_id} must contain binary voxel labels 0 and 1 only; "
                f"found {invalid_labels.tolist()}"
            )
        if image.shape[1:] != label.shape:
            raise ValueError(
                f"Image and mask shapes differ for {pair.image_id}: {image.shape[1:]} != {label.shape}"
            )
        if image.shape[0] != 1:
            raise ValueError(
                f"The inference UNet3D expects one image channel, got {image.shape[0]} for {pair.image_id}"
            )
        if self.classes is not None and (
            label.min() < 0 or label.max() >= self.classes
        ):
            raise ValueError(
                f"Mask labels for {pair.image_id} must be in [0, {self.classes - 1}]"
            )
        if self.normalize:
            low, high = float(image.min()), float(image.max())
            image = (image - low) / (high - low) if high > low else np.zeros_like(image)
        if self.patch_size:
            image, label = _crop_or_pad(
                image,
                label,
                self.patch_size,
                self.random_crop,
                self.foreground_probability,
            )
        if self.random_rotation_degrees > 0 or self.random_rotation_90_probability > 0:
            return _random_rotate(
                image,
                label,
                self.random_rotation_degrees,
                self.random_rotation_90_probability,
            )
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(
            np.ascontiguousarray(label)
        )


class NumorphDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.train_dataset = self.test_dataset = None
        self.validation_fold = None
        self._archive = None

    def prepare_data(self):
        path = Path(self.args["dataset_path"])
        if self.args.get("download_dataset"):
            download_dataset(
                self.args.get("dataset_url") or NUMORPH_DATASET_URL,
                path,
                overwrite=self.args.get("overwrite_dataset", False),
            )
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {path}. Use --download-dataset to fetch it."
            )

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return
        path = Path(self.args["dataset_path"])
        if path.is_file():
            if not zipfile.is_zipfile(path):
                raise ValueError(f"Dataset file is not a ZIP archive: {path}")
            self._archive = extract_dataset_archive(path)
            path = Path(self._archive.name)
        pairs = read_bia_pairs(path)
        requested_fold = self.args.get("validation_fold")
        if requested_fold is not None and requested_fold < 0:
            raise ValueError(
                "validation-fold must be 0 (random) or a positive one-based fold number"
            )
        requested_fold = (
            requested_fold - 1 if requested_fold and requested_fold > 0 else None
        )
        train, test, self.validation_fold = cross_validation_split(
            pairs,
            self.args.get("cross_validation_folds", 5),
            requested_fold,
            self.args["general_seed"],
        )
        for option in ("max_training_volumes", "max_validation_volumes"):
            limit = self.args.get(option)
            if limit is not None and limit < 1:
                raise ValueError(f"{option.replace('_', '-')} must be at least 1")
        train = train[: self.args.get("max_training_volumes")]
        test = test[: self.args.get("max_validation_volumes")]
        patch = tuple(int(value) for value in self.args["patch_size"].split(","))
        if len(patch) != 3 or any(value <= 0 or value % 4 for value in patch):
            raise ValueError(
                "patch-size must be three positive, comma-separated multiples of 4"
            )
        if self.args["patches_per_volume"] < 1:
            raise ValueError("patches-per-volume must be at least 1")
        if not 0.0 <= self.args["foreground_patch_probability"] <= 1.0:
            raise ValueError("foreground-patch-probability must be between 0 and 1")
        rotation_degrees = self.args.get("random_rotation_degrees", 2.0)
        if rotation_degrees < 0:
            raise ValueError("random-rotation-degrees must be non-negative")
        rotation_90_probability = self.args.get("random_rotation_90_probability", 0.5)
        if not 0 <= rotation_90_probability <= 1:
            raise ValueError("random-rotation-90-probability must be between 0 and 1")
        spacing_value = self.args.get("target_voxel_spacing", "3.0,1.0,1.0")
        spacing = tuple(
            float(value)
            for value in (
                spacing_value.split(",")
                if isinstance(spacing_value, str)
                else spacing_value
            )
        )
        if len(spacing) != 3 or any(value <= 0 for value in spacing):
            raise ValueError(
                "target-voxel-spacing must be three positive values in Z,Y,X order"
            )
        common = {
            "patch_size": patch,
            "normalize": self.args["normalize_input"],
            "classes": self.args["n_class"],
            "target_spacing": spacing,
        }
        self.train_dataset = VolumeDataset(
            train,
            random_crop=True,
            samples_per_volume=self.args["patches_per_volume"],
            foreground_probability=self.args["foreground_patch_probability"],
            random_rotation_degrees=rotation_degrees,
            random_rotation_90_probability=rotation_90_probability,
            **common,
        )
        self.test_dataset = VolumeDataset(test, random_crop=False, **common)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args["training_batch_size"],
            num_workers=self.args["num_workers"],
            shuffle=True,
            persistent_workers=self.args["num_workers"] > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args["test_batch_size"],
            num_workers=self.args["num_workers"],
            shuffle=False,
            persistent_workers=self.args["num_workers"] > 0,
        )

    def val_dataloader(self):
        return self.test_dataloader()
