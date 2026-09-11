"""Create image-readable TIFF samples without changing model test tensors.

BioImage.IO test tensors describe the complete model boundary, including the
batch dimension.  Sample TIFFs are presentation artifacts: they contain one
selected batch item in a conventional image-axis order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile

LAYOUTS = ("auto", "bcyx", "bczyx")
CHANNEL_POLICIES = ("squeeze-singleton", "preserve")


def _axis_letter(axis: Any) -> str:
    """Return the compact letter represented by one RDF axis declaration."""
    if isinstance(axis, str):
        value = axis
    elif isinstance(axis, dict):
        axis_type = axis.get("type")
        value = axis.get("id") if axis_type == "space" else axis_type
    else:
        raise ValueError(f"invalid RDF axis descriptor: {axis!r}")
    letter = {"batch": "B", "channel": "C"}.get(str(value).lower(), str(value).upper())
    if letter not in "BCZYX":
        raise ValueError(f"unsupported non-image RDF axis: {value!r}")
    return letter


def resolve_axes(axes: Sequence[Any]) -> str:
    """Resolve BioImage.IO axis descriptors to ``B``, ``C``, ``Z``, ``Y``, ``X``."""
    if not isinstance(axes, (list, tuple)) or not axes:
        raise ValueError("tensor descriptor must declare RDF axes")
    result = "".join(_axis_letter(axis) for axis in axes)
    if len(set(result)) != len(result):
        raise ValueError(f"RDF axes must be unique: {result}")
    if "X" not in result or "Y" not in result:
        raise ValueError("RDF axes must include both X and Y spatial axes")
    if "B" not in result or "C" not in result:
        raise ValueError("standard image-model axes require exactly one batch and channel axis")
    return result


def create_sample_tensor(
    array: np.ndarray,
    axes: Sequence[Any],
    destination: str | Path,
    *,
    layout: str = "auto",
    batch_index: int = 0,
    channel_policy: str = "squeeze-singleton",
) -> dict[str, Any]:
    """Write one RDF-described tensor batch as an image-readable TIFF.

    ``layout="auto"`` trusts the axis order declared in ``axes``.  An explicit
    layout additionally requires that declaration to be exactly ``BCYX`` or
    ``BCZYX``.  In either case, the stored image is reordered to ``YX``/``ZYX``
    or ``CYX``/``CZYX``.  Only the chosen batch axis and, when requested, a
    singleton channel axis are removed; values and spatial dimensions are kept.

    Returns a small record suitable for inclusion in run provenance.
    """
    array = np.asarray(array)
    resolved = resolve_axes(axes)
    layout = layout.lower()
    if layout not in LAYOUTS:
        raise ValueError(f"sample layout must be one of {', '.join(LAYOUTS)}")
    if channel_policy not in CHANNEL_POLICIES:
        raise ValueError(
            f"sample channel policy must be one of {', '.join(CHANNEL_POLICIES)}"
        )
    if array.ndim != len(resolved):
        raise ValueError(
            f"array rank {array.ndim} does not match {len(resolved)} RDF axes ({resolved})"
        )
    if layout != "auto" and resolved != layout.upper():
        raise ValueError(
            f"explicit layout {layout.upper()} does not match RDF axes {resolved}"
        )

    batch_axis = resolved.index("B")
    batch_size = array.shape[batch_axis]
    if batch_index < 0 or batch_index >= batch_size:
        raise IndexError(
            f"sample batch index {batch_index} is outside [0, {batch_size})"
        )
    # Selecting with take removes this axis alone, unlike a general squeeze.
    selected = np.take(array, batch_index, axis=batch_axis)
    remaining = resolved.replace("B", "")
    spatial_axes = "ZYX" if "Z" in resolved else "YX"
    target = "C" + spatial_axes
    # TIFF consumers receive one predictable order even if the RDF uses another.
    selected = np.transpose(selected, [remaining.index(axis) for axis in target])
    stored_axes = target
    if channel_policy == "squeeze-singleton" and selected.shape[0] == 1:
        selected = selected[0]
        stored_axes = spatial_axes

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        destination,
        selected,
        photometric="minisblack",
        metadata={"axes": stored_axes},
    )
    return {
        "model_axes": resolved,
        "tiff_axes": stored_axes,
        "batch_index": batch_index,
        "channel_policy": channel_policy,
        "filename": destination.name,
    }
