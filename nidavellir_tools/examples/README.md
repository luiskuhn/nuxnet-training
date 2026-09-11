# Model-package specification example

`model-package.example.yaml` is a structural reference for projects adopting
`nidavellir_tools`. Copy it into the consuming project's root, rename it as
appropriate, and replace every example value before building a package.

The axis list is also authoritative for presentation samples. The standalone
`create_sample_tensors.py --run-artifacts-dir <staging-directory>` command reads
each `test_tensor.source` and writes its `sample_tensor.source`; it is intended
for regenerating presentation TIFFs after training. Automatic layout writes
canonical `YX`/`ZYX` for a removed singleton channel or `CYX`/`CZYX` when
channels are retained. Batch selection is explicit, spatial dimensions are never
squeezed, and the exact `.npy` test tensors are never overwritten.

The example is intentionally not a runnable model definition: architecture,
tensor axes, preprocessing, postprocessing, dependencies, authorship, citation,
and licensing must describe the actual project and trained checkpoint. The
repository-level [`model-package.yaml`](../../model-package.yaml) is the concrete
NuxNet profile and demonstrates where project-specific values should live.
