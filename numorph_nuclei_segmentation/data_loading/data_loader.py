"""Load BIA/BioImage.IO OME-TIFF segmentation datasets."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import zipfile
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

_FILE_COLUMNS = ("filename", "file_name", "file_path", "filepath", "path", "uri", "file", "files")
_IMAGE_FILE_COLUMNS = (*_FILE_COLUMNS, "image")
_ANNOTATION_FILE_COLUMNS = (*_FILE_COLUMNS, "annotation", "mask", "label")
_IMAGE_ID_COLUMNS = ("image_id", "image_uuid", "id", "name")
_SOURCE_COLUMNS = ("image_id", "source_image_id", "source_image_uuid", "source_image", "image", "source")

NUMORPH_DATASET_URL = "https://drive.google.com/file/d/1nwLPXoWEsBb3wLNwXHY3L23UiO-5IIyn/view?usp=drive_link"


def _download_url(url: str) -> str:
    """Convert public Google Drive share links to their file-download form."""
    parsed = urlparse(url)
    if parsed.hostname not in {"drive.google.com", "www.drive.google.com"}:
        return url
    parts = [part for part in parsed.path.split("/") if part]
    file_id = parts[parts.index("d") + 1] if "d" in parts and parts.index("d") + 1 < len(parts) else None
    if file_id is None:
        file_id = parse_qs(parsed.query).get("id", [None])[0]
    if not file_id:
        raise ValueError(f"Could not find a Google Drive file ID in URL: {url}")
    query = urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return urlunparse(("https", "drive.usercontent.google.com", "/download", "", query, ""))


def download_dataset(url: str, destination: str | Path, *, overwrite: bool = False) -> Path:
    """Download a public dataset URL atomically, including Google Drive links."""
    destination = Path(destination).expanduser()
    if destination.exists() and not overwrite:
        if not destination.is_file():
            raise IsADirectoryError(f"Dataset download destination is a directory: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(_download_url(url), headers={"User-Agent": "nuxnet-training/1.0"})
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        with urlopen(request) as response, temporary.open("wb") as output:  # nosec B310 - public URL selected by user
            if response.headers.get_content_type() == "text/html":
                raise RuntimeError("Dataset URL returned HTML instead of a file; check that the link is public")
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
        return [{_normalise_key(str(key)): (value or "").strip() for key, value in row.items()} for row in reader]


def _column(row: dict[str, str], choices: Iterable[str], table: str) -> str:
    for choice in choices:
        if row.get(choice):
            return row[choice]
    raise ValueError(f"{table} must contain one of these populated columns: {', '.join(choices)}")


def _metadata_root(root: Path) -> Path:
    candidates = sorted(root.rglob("images.tsv"))
    matches = [path.parent for path in candidates if (path.parent / "annotations.tsv").is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one directory containing images.tsv and annotations.tsv below {root}; found {len(matches)}")
    return matches[0]


def _resolve_file(metadata_root: Path, filename: str) -> Path:
    """Resolve paths relative to the TSV folder or its submission-package root."""
    path = Path(filename)
    if path.is_absolute():
        return path
    local = metadata_root / path
    package_relative = metadata_root.parent / path
    return local if local.exists() or not package_relative.exists() else package_relative


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
        image_id = next((row[column] for column in _IMAGE_ID_COLUMNS if row.get(column)), None)
        image_id = image_id or (f"{group}:{pair_id}" if group and pair_id else filename)
        if image_id in image_index:
            raise ValueError(f"Duplicate image identifier in images.tsv: {image_id}")
        image = _resolve_file(metadata_root, filename)
        # BIA annotation tables in the wild reference either the stable image
        # identifier, the relative file path, or only the source basename.
        for reference in (image_id, filename, Path(filename).name):
            previous = image_index.get(reference)
            if previous and previous[0] != image_id:
                raise ValueError(f"Ambiguous image reference in images.tsv: {reference}")
            image_index[reference] = (image_id, image, row.get("split") or None, group)

    pairs: list[VolumePair] = []
    seen: set[str] = set()
    for row in annotations:
        source = _column(row, _SOURCE_COLUMNS, "annotations.tsv")
        filename = _column(row, _ANNOTATION_FILE_COLUMNS, "annotations.tsv")
        source_record = image_index.get(source) or image_index.get(Path(source).name)
        if source_record is None:
            raise ValueError(f"Annotation references unknown image identifier or path: {source}")
        image_id, image, split, group = source_record
        if image_id in seen:
            raise ValueError(f"More than one segmentation annotation for image: {image_id}")
        annotation = _resolve_file(metadata_root, filename)
        for kind, candidate in (("image", image), ("annotation", annotation)):
            if not candidate.is_file():
                raise FileNotFoundError(f"{kind.title()} file for {image_id} not found: {candidate}")
            if not candidate.name.lower().endswith((".ome.tif", ".ome.tiff")):
                raise ValueError(f"{kind.title()} file must be OME-TIFF: {candidate}")
        pairs.append(VolumePair(image_id, image, annotation, split, group))
        seen.add(image_id)
    if not pairs:
        raise ValueError("No image/annotation pairs found in the BIA metadata tables")
    return pairs


def _read_ome(path: Path, *, mask: bool) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        if not tif.is_ome:
            raise ValueError(f"Expected OME-TIFF metadata in {path}")
        array = tif.series[0].asarray()
        axes = tif.series[0].axes.upper()
    for axis in "ST":
        if axis in axes:
            index = axes.index(axis)
            if array.shape[index] != 1:
                raise ValueError(f"{path} has non-singleton {axis}; export each scene/timepoint as a separate record")
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
    return np.transpose(array, tuple(axes.index(axis) for axis in wanted))


def _crop_or_pad(
    image: np.ndarray,
    label: np.ndarray,
    patch: tuple[int, int, int],
    random_crop: bool,
    foreground_probability: float,
):
    padding = [(0, 0)] + [(0, max(0, size - actual)) for actual, size in zip(label.shape, patch)]
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


def _random_rotate(image: np.ndarray, label: np.ndarray, max_degrees: float):
    """Rotate an image/mask pair about a random 3-D axis.

    The same sampling grid is used for both arrays. Intensities use trilinear
    interpolation, while class IDs use nearest-neighbour interpolation so that
    augmentation cannot create new mask classes.
    """
    axis = torch.randn(3, dtype=torch.float64)
    axis /= axis.norm().clamp_min(torch.finfo(axis.dtype).eps)
    angle = torch.empty((), dtype=torch.float64).uniform_(-max_degrees, max_degrees)
    angle = torch.deg2rad(angle)
    x, y, z = axis
    zero = torch.zeros((), dtype=axis.dtype)
    cross = torch.stack((torch.stack((zero, -z, y)), torch.stack((z, zero, -x)), torch.stack((-y, x, zero))))
    rotation = torch.eye(3, dtype=axis.dtype) * torch.cos(angle)
    rotation += (1 - torch.cos(angle)) * axis[:, None] * axis[None, :]
    rotation += torch.sin(angle) * cross

    # affine_grid coordinates are X,Y,Z-normalised. Conjugating by the
    # dimensions makes ``rotation`` a rotation in voxel coordinates even for
    # the deliberately anisotropic 128x128x32 patch.
    dimensions = torch.tensor(label.shape[::-1], dtype=axis.dtype)
    rotation = rotation * dimensions[None, :] / dimensions[:, None]
    theta = torch.cat((rotation, torch.zeros((3, 1), dtype=axis.dtype)), dim=1).to(torch.float32)[None]
    image_tensor = torch.from_numpy(np.ascontiguousarray(image))[None]
    label_tensor = torch.from_numpy(np.ascontiguousarray(label)).to(torch.float32)[None, None]
    grid = F.affine_grid(theta, image_tensor.shape, align_corners=False)
    image_tensor = F.grid_sample(image_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    label_tensor = F.grid_sample(label_tensor, grid, mode="nearest", padding_mode="zeros", align_corners=False)
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
    ):
        self.pairs = list(pairs)
        self.patch_size = tuple(patch_size) if patch_size else None
        self.random_crop = random_crop
        self.normalize = normalize
        self.classes = classes
        self.samples_per_volume = samples_per_volume
        self.foreground_probability = foreground_probability
        self.random_rotation_degrees = float(random_rotation_degrees)

    def __len__(self):
        return len(self.pairs) * self.samples_per_volume

    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        image = _read_ome(pair.image, mask=False).astype(np.float32, copy=False)
        label = _read_ome(pair.annotation, mask=True)
        if not np.issubdtype(label.dtype, np.integer):
            if not np.array_equal(label, label.astype(np.int64)):
                raise ValueError(f"Mask contains non-integer class labels: {pair.annotation}")
        label = label.astype(np.int64, copy=False)
        invalid_labels = np.setdiff1d(np.unique(label), (0, 1))
        if invalid_labels.size:
            raise ValueError(
                f"Mask for {pair.image_id} must contain binary voxel labels 0 and 1 only; "
                f"found {invalid_labels.tolist()}"
            )
        if image.shape[1:] != label.shape:
            raise ValueError(f"Image and mask shapes differ for {pair.image_id}: {image.shape[1:]} != {label.shape}")
        if image.shape[0] != 1:
            raise ValueError(f"The inference UNet3D expects one image channel, got {image.shape[0]} for {pair.image_id}")
        if self.classes is not None and (label.min() < 0 or label.max() >= self.classes):
            raise ValueError(f"Mask labels for {pair.image_id} must be in [0, {self.classes - 1}]")
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
        if self.random_rotation_degrees > 0:
            return _random_rotate(image, label, self.random_rotation_degrees)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(label))


class NumorphDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.train_dataset = self.test_dataset = None
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
            raise FileNotFoundError(f"Dataset path does not exist: {path}. Use --download-dataset to fetch it.")

    def setup(self, stage=None):
        path = Path(self.args["dataset_path"])
        if path.is_file():
            if not zipfile.is_zipfile(path):
                raise ValueError(f"Dataset file is not a ZIP archive: {path}")
            self._archive = extract_dataset_archive(path)
            path = Path(self._archive.name)
        pairs = read_bia_pairs(path)
        train = [pair for pair in pairs if (pair.split or "").lower() == "train"]
        test = [pair for pair in pairs if (pair.split or "").lower() in {"test", "validation", "val"}]
        if not train and not test:
            generator = torch.Generator().manual_seed(self.args["general_seed"])
            train, test = [], []
            groups = sorted({pair.group or "all" for pair in pairs})
            for group in groups:
                members = [pair for pair in pairs if (pair.group or "all") == group]
                order = torch.randperm(len(members), generator=generator).tolist()
                test_size = max(1, round(len(members) * self.args["test_percent"])) if len(members) > 1 else 0
                test.extend(members[index] for index in order[:test_size])
                train.extend(members[index] for index in order[test_size:])
        elif len(train) + len(test) != len(pairs):
            raise ValueError("Every record must use split train, validation/val, or test")
        patch = tuple(int(value) for value in self.args["patch_size"].split(","))
        if len(patch) != 3 or any(value <= 0 or value % 4 for value in patch):
            raise ValueError("patch-size must be three positive, comma-separated multiples of 4")
        if self.args["patches_per_volume"] < 1:
            raise ValueError("patches-per-volume must be at least 1")
        if not 0.0 <= self.args["foreground_patch_probability"] <= 1.0:
            raise ValueError("foreground-patch-probability must be between 0 and 1")
        rotation_degrees = self.args.get("random_rotation_degrees", 2.0)
        if rotation_degrees < 0:
            raise ValueError("random-rotation-degrees must be non-negative")
        common = {"patch_size": patch, "normalize": self.args["normalize_input"], "classes": self.args["n_class"]}
        self.train_dataset = VolumeDataset(
            train,
            random_crop=True,
            samples_per_volume=self.args["patches_per_volume"],
            foreground_probability=self.args["foreground_patch_probability"],
            random_rotation_degrees=rotation_degrees,
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
