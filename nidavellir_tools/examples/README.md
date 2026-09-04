# Model-package specification example

`model-package.example.yaml` is a structural reference for projects adopting
`nidavellir_tools`. Copy it into the consuming project's root, rename it as
appropriate, and replace every example value before building a package.

The example is intentionally not a runnable model definition: architecture,
tensor axes, preprocessing, postprocessing, dependencies, authorship, citation,
and licensing must describe the actual project and trained checkpoint. The
repository-level [`model-package.yaml`](../../model-package.yaml) is the concrete
NuxNet profile and demonstrates where project-specific values should live.
