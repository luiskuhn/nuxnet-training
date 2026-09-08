Usage
=====

The top-level :doc:`readme` is the maintained guide for Docker, Conda, MLflow,
dataset preparation, hyperparameters, and example runs. Keeping the executable
commands in one place prevents this Sphinx site from drifting from the CLI.

To inspect every option supported by the installed version, run:

.. code-block:: bash

   python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation --help

For a quick CPU run, provide an extracted dataset and limit the run explicitly:

.. code-block:: bash

   python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
       --dataset-path data/NUMORPH_SEM_SEG_DATASET \
       --accelerator cpu --devices 1 --max_epochs 2 --test-epochs 1

This smoke test verifies execution only; it is not suitable for comparing model
quality. The README's hyperparameter table explains the defaults used for a full
training run.
