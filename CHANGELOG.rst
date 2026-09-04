==========
Changelog
==========

This project adheres to `Semantic Versioning <https://semver.org/>`_.

Unreleased
----------

**Added**

* Select checkpoints and schedule learning rate from foreground validation IoU.
* Aggregate IoU and accuracy from epoch-level, DDP-reduced confusion counts.
* Evaluate complete held-out volumes with deterministic sliding-window inference.

* A domain-independent model registry tool for verified staging from local,
  HTTP, MLflow, or Hugging Face sources, PyTorch loading, BioImage.IO validation,
  and complete-package Hugging Face publication.
* A review of the NuxNet exporter boundary, proposed reusable FAIR package
  architecture, and operational acceptance checklist.
* Symmetric parent-model loading and child export, including metadata sidecars,
  strict transfer-learning initialization, refreshed test evidence/model cards,
  and explicit parent lineage across repeated training cycles.
* Move the package builder and lifecycle registry into ``nidavellir_tools``;
  make initial package construction declarative and independent of NuxNet,
  microscopy, tensor dimensionality, and vision task semantics.
* Add a README quick start with setup, build, inspect, checksum, staging, loading,
  validation, publication, and local/HTTP/Hugging Face/MLflow source examples.
* Add an annotated, domain-neutral model-package specification under
  ``nidavellir_tools/examples`` while retaining the root NuxNet profile separately.
* Atomic public dataset downloads, including direct support for the NuMorph
  Google Drive share link.
* Reviewed dataset documentation covering its biological source, resolution
  groups, volume dimensions, annotation semantics, and submission status.
* Converted the project README from reStructuredText to Markdown.
* Added configurable, seeded cross-validation folds for training evaluation.

**Fixed**

* Prevented multi-GPU training from hanging between epochs by spawning data
  loader workers instead of forking them after CUDA initialization.
* Removed redundant validation/test inference work by making complete,
  non-overlapping sliding-window coverage the default. Overlap remains
  configurable when tile-edge blending is required.
* Aligned CI, documentation, and container workflows with the supported
  Python 3.12 environment and pinned dependencies.
* Made the container command overridable by MLflow and added dependency smoke
  tests, a health check, and a reduced Docker build context.
* Expanded the README with compact CPU/GPU training commands and scientific
  descriptions of the NuMorph task, 3D U-Net, focal loss, and reported metrics.


1.0.0 (2021-03-28)
------------------

**Added**

* Initial implementation of the predecessor 3D U-Net
* Several runs conducted for the mlf-core paper

**Fixed**

**Dependencies**

**Deprecated**
