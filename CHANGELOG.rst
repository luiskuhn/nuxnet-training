==========
Changelog
==========

This project adheres to `Semantic Versioning <https://semver.org/>`_.

Unreleased
----------

**Added**

* Atomic public dataset downloads, including direct support for the NuMorph
  Google Drive share link.
* Reviewed dataset documentation covering its biological source, resolution
  groups, volume dimensions, annotation semantics, and submission status.
* Converted the project README from reStructuredText to Markdown.
* Added configurable, seeded cross-validation folds for training evaluation.

**Fixed**

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
