FAIR model packages and registries
==================================

Design and scope
----------------

The commands in ``nidavellir_tools`` form a reusable package lifecycle layer.
``build_model_package.py`` builds a package from a project-owned BioImage.IO RDF
specification, checkpoint, model card, test evidence, and optional provenance.
``model_package_registry.py`` stages and verifies packages, loads models together
with metadata, exports lineage-linked children, validates BioImage.IO archives,
and publishes complete Hugging Face repositories.

Neither command imports the application package or assumes a microscopy task,
spatial dimensionality, axis order, class vocabulary, normalization, or output
activation.  Those scientifically meaningful choices belong in the RDF and model
card.  The shared implementation targets projects using PyTorch, Lightning,
MLflow, NumPy, PyYAML, and the same checkpoint/package conventions.

The upstream ``nidavellir-tools`` example specification is the annotated,
domain-neutral structural reference shipped with the tools.  Consumers copy it
into their project and replace every example value.  The repository-root
``model-package.yaml`` remains separate because it is the concrete NuxNet
profile; keeping application metadata out of the reusable directory prevents an
example from becoming an accidental default.

Implemented reusable layer
--------------------------

The installed ``nidavellir`` registry interface is deliberately independent from any application domain.  It treats the BioImage.IO RDF as the portable package contract and
provides these operations:

.. code-block:: bash

   # Local directory/ZIP or HTTP URL
   nidavellir stage model.zip .model-cache/example

   # Immutable Hub revision, or an MLflow run artifact
   nidavellir stage hf://owner/model .model-cache/example --revision COMMIT
   nidavellir stage mlflow://RUN_ID/model .model-cache/example

   nidavellir inspect .model-cache/example
   nidavellir load .model-cache/example \
       --representation pytorch_state_dict \
       --weights-output work/parent.pt --metadata-output work/parent.json
   nidavellir validate .model-cache/example --report reports/example.json
   nidavellir publish-hf .model-cache/example owner/model

Staging accepts one RDF package, rejects ZIP traversal, copies it to a stable
destination, and checks every local artifact for which the RDF declares a
SHA-256 digest.  Loading prefers TorchScript because it does not require project
source.  ``--representation pytorch_state_dict`` reconstructs an architecture
from an RDF source file and callable (or an importable ``module:callable``),
applies its ``kwargs``, and strictly loads the state dictionary.

Transfer-learning round trip
----------------------------

Loading and export use the same RDF contract.  ``load`` can materialize both a
tensor-only state dictionary and a JSON sidecar containing the complete RDF,
provenance, and selected representation.  This project accepts those artifacts
as initialization, verifies that their SHA-256 values match, and records the
metadata in the new MLflow run.  The two training options must be supplied
together:

.. code-block:: bash

   python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
       --initial-weights work/parent.pt --parent-metadata work/parent.json \
       --dataset-path /new-data

After training, ``export-child`` takes the parent package as the canonical tensor
and architecture specification.  It strictly checks the new checkpoint against
that architecture, removes stale alternative executable representations, updates
weight/documentation/test-output hashes, and records explicit parent lineage:

.. code-block:: bash

   nidavellir export-child .model-cache/example \
       lightning_logs/checkpoints/best.ckpt output/child \
       --state-dict-key state_dict --strip-prefix model. --version 2.0.0 \
       --parent-identifier hf://owner/model@COMMIT \
       --test-output work/new-test-output.npy --model-card work/README.md

The new test output must be generated from the parent's declared test input using
the newly trained model and the RDF-declared processing.  A new model card is
mandatory because scientific results, dataset, intended use, and limitations may
have changed.  Requiring both files prevents a technically plausible child from
silently retaining stale claims or expected outputs.  The resulting child can be
passed back to ``stage`` and ``load`` for the next cycle.  ``export-child`` also
writes a sibling ZIP containing the same verified package for BioImage.IO
validation and submission; publish the directory itself to Hugging Face.

The Hugging Face and MLflow integrations use their normal authentication and
cache configuration.  Install ``huggingface_hub`` only for ``hf://`` staging or
``publish-hf``; MLflow is only needed for ``mlflow://``.  Local use has no new
runtime dependency.  ``publish-hf`` creates a model repository if necessary and
uploads the complete, verified directory rather than weights alone.  BioImage.IO
submission remains intentionally review-gated: with Nidavellir 0.3.0,
``validate`` calls the official ``bioimageio.core.test_description`` API and
returns structured results. After successful checks and scientific review, the ZIP
is submitted through the Zoo's supported upload/review workflow.

Structured official validation (requires Nidavellir 0.3.0)
---------------------------------------------------------

This feature requires the upcoming ``nidavellir-tools==0.3.0`` release containing
the structured validation API. It is not available in 0.2.0. Install the optional
extra in the packaging/validation environment, preserving NuxNet's requirements:

.. code-block:: bash

   python -m pip install -r requirements.txt "nidavellir-tools[bioimageio]==0.3.0"
   python -m pip check
   nidavellir validate output/child.zip --report reports/child.json

``inspect`` checks declared artifact hashes, while the official validator checks
metadata and runs inference tests against the declared model contract. Defaults
are CPU, ``pytorch_state_dict`` weights, and the currently active environment.
Only ``passed`` is successful. ``failed`` describes integrity/official check
failures; ``error`` describes input, dependency, or execution problems. Both yield
a nonzero CLI exit status. Reports retain official diagnostics, package digest,
execution settings, and library versions.

Use fresh external report paths: existing files cannot be overwritten. A directory
digest and a ZIP-byte digest are different identities. Reports should remain
outside the package; modifying artifacts requires fresh validation. Mount reports
on a writable host directory when using disposable containers. The root README
provides a separate validation image that installs the extra without changing
the training image.

Optionally gate package construction before ZIP creation:

.. code-block:: bash

   nidavellir build --run-artifacts-dir run/model-package-inputs \
       --output-dir output/validated-child --validate-bioimageio \
       --validation-report reports/validated-child.json

On failure, the unpacked directory and report remain but no new ZIP is created.
An old ZIP is not removed; use fresh destinations. ``export-child`` and
``publish-hf`` do not invoke official validation automatically. Run checks on the
final parent and child artifacts before publication.

Only trusted models should be validated: their architecture code executes, and
referenced resources can require network access. Testing uses a temporary copy,
not a security sandbox. Current-environment validation does not demonstrate
environment recreatability, GPU compatibility, or scientific accuracy.

Full solution and boundaries
----------------------------

The resulting separation should remain explicit:

1. **Project-owned RDF specification and model card** record truthful tensor
   semantics, preprocessing, postprocessing, architecture, test fixtures,
   scientific validation, licensing, and provenance.  The generic builder never
   invents these domain claims.
2. **Portable package contract** is ``rdf.yaml`` plus only relative, hashed
   artifacts.  TorchScript is the preferred executable representation here;
   state dictionaries support inspection and fine-tuning when architecture code
   is available.
3. **Generic lifecycle tool** is ``model_package_registry.py``.  It stages from local,
   HTTP, MLflow, or Hugging Face, verifies integrity, loads supported PyTorch
   forms with their metadata, exports lineage-linked children using the same RDF
   contract, invokes BioImage.IO validation, and publishes complete folders.
4. **Repository governance** stays outside the transport.  Tokens come from the
   official clients, immutable revisions should be used in production, and
   publication requires human review of licensing, personally identifying data,
   citations, metrics, intended use, and limitations.

The builder is already configuration-driven: each project keeps its RDF template
and referenced architecture/dependency assets outside ``nidavellir_tools``.  New
representations such as ONNX or MLflow ``pyfunc`` should be independent adapters;
they must preserve this same package and lineage contract rather than add domain
logic to the lifecycle commands.

Operational acceptance checklist
--------------------------------

* Pin a source revision and retain checkpoint, source, dataset, and environment
  identifiers in provenance.
* Verify declared artifact hashes and run ``nidavellir validate`` on the final
  package. Require ``passed`` and retain the external JSON report with its digest.
* Compare loaded-model output with the exported test fixture using declared
  preprocessing/postprocessing and tolerances.
* Report held-out scientific metrics separately from technical execution tests.
* Confirm model, code, data, and cover licenses; scan the staged directory for
  secrets and sensitive data before publication.
* Upload the whole package, test a clean pull by immutable Hub revision, and only
  then submit the same validated package to the BioImage.IO review workflow.
