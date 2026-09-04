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

``nidavellir_tools/examples/model-package.example.yaml`` is the annotated,
domain-neutral structural reference shipped with the tools.  Consumers copy it
into their project and replace every example value.  The repository-root
``model-package.yaml`` remains separate because it is the concrete NuxNet
profile; keeping application metadata out of the reusable directory prevents an
example from becoming an accidental default.

Implemented reusable layer
--------------------------

``nidavellir_tools/model_package_registry.py`` is deliberately independent from any application domain.  It treats the BioImage.IO RDF as the portable package contract and
provides these operations:

.. code-block:: bash

   # Local directory/ZIP or HTTP URL
   python nidavellir_tools/model_package_registry.py stage model.zip .model-cache/example

   # Immutable Hub revision, or an MLflow run artifact
   python nidavellir_tools/model_package_registry.py stage hf://owner/model .model-cache/example --revision COMMIT
   python nidavellir_tools/model_package_registry.py stage mlflow://RUN_ID/model .model-cache/example

   python nidavellir_tools/model_package_registry.py inspect .model-cache/example
   python nidavellir_tools/model_package_registry.py load .model-cache/example \
       --representation pytorch_state_dict \
       --weights-output work/parent.pt --metadata-output work/parent.json
   python nidavellir_tools/model_package_registry.py validate .model-cache/example
   python nidavellir_tools/model_package_registry.py publish-hf .model-cache/example owner/model

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

   python nidavellir_tools/model_package_registry.py export-child .model-cache/example \
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
submission remains intentionally review-gated: ``validate`` invokes the official
``bioimageio test`` command, after which the ZIP is submitted through the Zoo's
supported upload/review workflow.

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
* Verify all local RDF artifacts and run ``bioimageio test`` on the final package.
* Compare loaded-model output with the exported test fixture using declared
  preprocessing/postprocessing and tolerances.
* Report held-out scientific metrics separately from technical execution tests.
* Confirm model, code, data, and cover licenses; scan the staged directory for
  secrets and sensitive data before publication.
* Upload the whole package, test a clean pull by immutable Hub revision, and only
  then submit the same validated package to the BioImage.IO review workflow.
