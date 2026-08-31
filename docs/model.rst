NuMorph nucleus-marker model
============================

The model trains the same reduced 3D U-Net architecture consumed by
``nuxnet-inference``. It predicts two semantic classes: background and
nucleus-marker voxels.

Dataset
-------

The NuMorph dataset contains 32 paired OME-TIFF image and binary marker-mask
volumes split evenly between the C075 and C121 physical-resolution groups.
Inputs are single-channel ``float32`` arrays and masks are ``uint8`` arrays
with values ``{0, 1}``.

Training
--------

Training uses deterministic, resolution-stratified holdout selection and
foreground-aware ``16x64x64`` patches. Min/max normalization matches the
inference preprocessing. The final ``numorph_unet3d.pt`` artifact is a plain
PyTorch state dictionary suitable for the inference CLI with ``--classes 2``.
