---
library_name: pytorch
tags:
  - bioimage.io
---

# BioImage.IO model

This repository contains a trained PyTorch model packaged according to the
BioImage.IO model resource description in `rdf.yaml`.

## Validation

Run `bioimageio test <model-package.zip>` to validate the package format,
reload the exported PyTorch state dictionary, and reproduce the packaged test
output. Scientific validation should use previously unseen complete volumes and
report foreground IoU (`IoU₁`) and mean IoU. The technical test tensors are not
a substitute for independent biological validation.
