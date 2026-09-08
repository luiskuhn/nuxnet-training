<!-- markdownlint-disable MD013 -->

# nuxnet-training

![Graphical abstract of the nuxnet-training workflow](docs/images/graph_abstract_nuxnet_training.png)

Reproducible 3D U-Net training for nucleus-marker detection in NuMorph light-sheet microscopy data. The exported PyTorch state dictionary is compatible with [`nuxnet-inference`](https://github.com/luiskuhn/nuxnet-inference).

## Contents

- [What this repository trains](#what-this-repository-trains)
- [Data handling](#image-and-mask-loading)
- [Quick start with Docker](#quick-start-with-docker)
- [Hyperparameters](#hyperparameters)
- [Example configurations](#example-configurations)
- [Dataset format and splitting](#dataset-format-and-splitting)
- [Conda and MLflow](#other-ways-to-run)
- [Outputs and compatibility](#outputs-and-compatibility)
- [FAIR model packaging](#fair-model-packaging-and-transfer-learning)
- [Development and documentation](#development-and-documentation)

## What this repository trains

The model is a compact residual 3D U-Net with 32-, 64-, and 128-channel feature levels. Two stride-2 convolutions form the encoder, and two nearest-neighbor upsampling stages restore the original resolution while reusing encoder features through skip connections. Every two-convolution block has a residual shortcut (a normalized 1×1×1 projection when channels change), and the final 1×1×1 head emits two-channel background/marker logits. Spatial dropout acts only on learned feature maps, never directly on the raw input.

Training uses unweighted cross-entropy plus foreground soft Dice and Adam. Checkpoints and learning-rate scheduling use foreground validation IoU, while reporting loss, voxel accuracy, per-class IoU and mean IoU. Because the foreground is sparse, prefer nucleus-class and mean IoU over accuracy when comparing runs. The architecture remains below 2.5 million parameters and retains the compact model's memory profile.

The default `NUMORPH_SEM_SEG_DATASET` contains 32 paired OME-TIFF image/mask volumes: 16 at 0.75 × 0.75 × 2.5 µm and 16 at 1.21 × 1.21 × 4.0 µm. Images are normalized, single-channel `float32`; masks are binary `uint8`.

> [!IMPORTANT]
> Masks are manually curated **2D middle-Z nucleus markers for detection**, stored in 3D volumes. They are not complete 3D boundaries or instance segmentations. Confirm the dataset's reuse terms before redistribution because its packaged metadata currently records the license and public resource identifiers as pending.

## Image and mask loading

The loader reads the first series from each OME-TIFF and uses the OME `PhysicalSizeZ`, `PhysicalSizeY`, and `PhysicalSizeX` values as the **physical voxel size**. Values are converted to µm and kept in **Z,Y,X** order. Images are returned as `CZYX`; masks are returned as `ZYX`. Singleton scene and time axes are removed, while files containing multiple scenes or time points are rejected so that each table row always describes one volume.

### Resampling to a common physical voxel size

Before selecting patches, every image/mask pair is placed on the physical voxel-size grid requested by `--target-voxel-spacing` (default `3.0,1.0,1.0` µm per voxel). The output length of each axis is `input length × source voxel size / target voxel size`, preserving the volume's physical extent. Images use Gaussian anti-aliasing when an axis is reduced and trilinear interpolation; masks use nearest-neighbor interpolation. A component-centroid safeguard keeps sparse, one-slice markers from disappearing during downsampling. The resampled volumes are cached in memory and reused for later patches.

### Patch sampling and augmentation

The default patch is `32,128,128` voxels, representing approximately `96 × 128 × 128` µm on the default grid. Each training volume contributes eight patches per epoch. A patch is centered on a randomly selected foreground voxel with probability `0.8`; otherwise its start is random. Volumes smaller than a patch are zero-padded. Validation and test samples use one deterministic center patch per volume.

Training patches may receive an exact random 0°, 90°, 180°, or 270° rotation in the XY plane, followed by a continuous angle sampled from −10° to +10°. Image and mask transformations stay aligned, mask interpolation remains nearest-neighbor, and Z is never mixed with a lateral axis. There are no random flips, intensity perturbations, elastic transforms, or scale jitter. Per-volume min/max image normalization is enabled by default.

## Logits, loss, and probabilities

The U-Net head returns raw two-channel logits. Training uses Dice plus cross-entropy, computed as

```text
loss = ce_loss_weight × cross_entropy(logits, target)
     + dice_loss_weight × foreground_soft_dice_loss(logits, target)
```

Cross-entropy receives the raw logits. Foreground Dice applies softmax internally, selects class 1, and aggregates intersection and denominator over the complete batch and all spatial dimensions with smoothing for finite empty-foreground behavior. Equal `--class-weights 1.0,1.0` means ordinary unweighted cross-entropy; non-uniform weights are an explicit experiment.

Metrics and packaged BioImage.IO inference apply softmax explicitly when probabilities are needed. Plain `UNet3D` state-dictionary and TorchScript consumers receive logits and must apply `softmax(dim=1)` themselves. They must also resample source OME data to the target Z,Y,X grid before tensor inference; the exported provenance and `rdf.yaml` record that grid but the tensor model cannot infer source voxel calibration from a tensor alone.

## Quick start with Docker

Docker is recommended because it provides pinned Python and CUDA dependencies. Build the image, create persistent output directories, and train on an extracted dataset:

### Prerequisites

- Docker for the recommended container workflow.
- An extracted dataset directory (or writable location for an automatic download).
- For GPU training, an NVIDIA driver compatible with CUDA 12.4 and the NVIDIA Container Toolkit.

```bash
mkdir -p "$PWD/mlruns" "$PWD/output"
docker build -t nuxnet-training .

docker run --rm \
  -v "$PWD/dataset:/data:ro" \
  -v "$PWD/mlruns:/mlruns" \
  nuxnet-training \
  --dataset-path /data --max_epochs 100 --accelerator cpu --devices 1
```

`--dataset-path` also accepts a ZIP. To download the public dataset automatically, mount a writable directory and use:

```bash
docker run --rm \
  -v "$PWD/data:/data" -v "$PWD/mlruns:/mlruns" \
  nuxnet-training \
  --download-dataset --dataset-path /data/NUMORPH_SEM_SEG_DATASET.zip \
  --max_epochs 100 --accelerator cpu --devices 1
```

For an NVIDIA GPU, install the NVIDIA Container Toolkit and add `--gpus all` to `docker run`, then pass `--accelerator gpu --devices 1`. The best inference-compatible model is written to the mounted `mlruns/numorph_unet3d.pt`; Lightning checkpoints and TensorBoard events are stored below the same directory.

## Hyperparameters

These options have the greatest effect on training quality, runtime, and memory:

| Hyperparameter | CLI flag | Default | Description |
| --- | --- | ---: | --- |
| Epochs | `--max_epochs` | `1000` | Maximum training epochs. |
| Learning rate | `--lr` | `0.0001` | Initial Adam learning rate. |
| LR reduction factor | `--lr-scheduler-factor` | `0.5` | Multiply the learning rate by this factor when foreground validation IoU plateaus. |
| LR scheduler patience | `--lr-scheduler-patience` | `5` | Validation evaluations without sufficient improvement tolerated before reducing the rate. With validation every 5 epochs, patience 6 reduces after roughly 35 stagnant epochs because PyTorch reduces after the seventh non-improving evaluation. |
| LR improvement threshold | `--lr-scheduler-threshold` | `1e-5` | Minimum absolute improvement in foreground validation IoU that resets scheduler patience. |
| LR scheduler cooldown | `--lr-scheduler-cooldown` | `2` | Validation evaluations to wait after a rate reduction before resuming plateau checks. |
| Minimum learning rate | `--min-lr` | `1e-6` | Lower bound for learning-rate reductions. |
| Patch size | `--patch-size Z,Y,X` | `32,128,128` | Spatial context per sample. Each dimension must be divisible by four; larger patches require more memory. |
| Target physical voxel size | `--target-voxel-spacing Z,Y,X` | `3.0,1.0,1.0` | OME-style physical size of one output voxel in Z,Y,X order, in µm per voxel. |
| Training batch size | `--training-batch-size` | `1` | Patches per optimizer step. Increase only when memory permits. |
| Patches per volume | `--patches-per-volume` | `8` | Random training patches drawn from each volume per epoch. Higher values provide more sampling at greater runtime. |
| Foreground sampling | `--foreground-patch-probability` | `0.8` | Probability that a training patch is centered on a marker (`0`–`1`). |
| Loss weights | `--ce-loss-weight`, `--dice-loss-weight` | `1.0`, `1.0` | Contributions to the combined Dice+CE objective. |
| Class weights | `--class-weights` | `1.0,1.0` | Optional explicit per-class weights; equal values give unweighted cross-entropy. |
| Dropout | `--dropout-rate` | `0.10` | Probability for all 3D dropout layers: two per residual convolution block and one after each down/up transition. |
| Rotation | `--random-rotation-degrees` | `10.0` | Maximum continuous XY training rotation; Z is never mixed with lateral axes. |
| Right-angle rotation | `--random-rotation-90-probability` | `0.5` | Probability of an exact random 0°, 90°, 180°, or 270° XY rotation. |
| Folds | `--cross-validation-folds` | `5` | Number of shuffled, resolution-group-stratified folds (at least two). |
| Held-out fold | `--validation-fold` | `0` | One-based validation/test fold. `0` selects reproducibly from the general seed. |
| Validation interval | `--test-epochs` | `10` | Run validation every positive N epochs and observe `val_iou_1` for the plateau scheduler at the same frequency. Scheduler patience and cooldown count validation evaluations, not training epochs. For example, `--test-epochs 5 --lr-scheduler-patience 6` reduces the LR after approximately 35 stagnant training epochs because PyTorch reduces after the seventh non-improving validation observation. Final testing always uses the best checkpoint. |
| Input normalization | `--normalize-input` / `--no-normalize-input` | enabled | Per-volume min/max normalization compatible with inference. Disable only for prepared inputs. |

Reproducibility and execution options:

| CLI flag | Default | Description |
| --- | ---: | --- |
| `--general-seed`, `--pytorch-seed` | `0`, `0` | Python/NumPy and PyTorch random seeds. |
| `--accelerator` | `auto` | Lightning accelerator: `auto`, `cpu`, or `gpu`. |
| `--devices` | `auto` | Device count or `auto`; use with `--strategy` for distributed training. |
| `--num_workers` | `0` | Data-loader workers. Increase when input loading limits throughput; positive values retain safe spawn-based workers. |
| `--log-interval` | `100` | Training steps between log writes. |
| `--n-channels`, `--n-class` | `1`, `2` | Input channels and output classes. Defaults match NuMorph and inference. |
| `--test-batch-size` | `1` | Number of sliding inference windows processed simultaneously during validation/test. |
| `--inference-overlap` | `0.0` | Sliding-window overlap fraction in `[0,1)` for validation/test. Non-overlapping windows cover every voxel with the least computation; increase this only when reducing tile-edge seams is worth the additional 3-D inference work. A `0.5` overlap evaluates roughly eight times as many interior windows. |
| `--max-training-volumes`, `--max-validation-volumes` | unset | Limit volumes after splitting for smoke tests only. |

Dataset options are `--dataset-path`, `--download-dataset`, `--dataset-url`, and `--overwrite-dataset`. Run `docker run --rm nuxnet-training --help` (or the Python command below with `--help`) for the authoritative complete CLI.

### Monte Carlo dropout inference

The core `UNet3D` returns logits. For repeated stochastic inference passes, call
`enable_mc_dropout(model)` after loading weights. The helper first puts the whole
model in evaluation mode, then enables only `Dropout3d`; `BatchNorm3d` remains in
evaluation mode so its running statistics are not updated:

```python
from numorph_nuclei_segmentation.model import enable_mc_dropout

enable_mc_dropout(model)
samples = [model(volume).softmax(dim=1) for _ in range(20)]
```

Aggregate those samples in downstream inference code as appropriate. The helper
intentionally does not prescribe an uncertainty metric or aggregation strategy.

## Example configurations

### Production GPU run

This retains the recommended patch and sampling defaults while making the fold explicit:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/data:ro" -v "$PWD/mlruns:/mlruns" \
  nuxnet-training \
  --dataset-path /data --accelerator gpu --devices 1 \
  --max_epochs 1000 --lr 0.0001 \
  --patch-size 32,128,128 --training-batch-size 1 \
  --patches-per-volume 8 --foreground-patch-probability 0.8 \
  --target-voxel-spacing 3.0,1.0,1.0 \
  --random-rotation-degrees 10.0 --random-rotation-90-probability 0.5 \
  --ce-loss-weight 1.0 --dice-loss-weight 1.0 \
  --class-weights 1.0,1.0 --validation-fold 1
```

### Fast end-to-end smoke test

This verifies download, loading, training, validation, testing and export; its small sample and patches do **not** measure model quality:

```bash
docker run --rm \
  -v "$PWD/data:/data" -v "$PWD/mlruns:/mlruns" \
  nuxnet-training \
  --download-dataset --dataset-path /data/NUMORPH_SEM_SEG_DATASET.zip \
  --accelerator cpu --devices 1 --max_epochs 2 --test-epochs 1 \
  --max-training-volumes 2 --max-validation-volumes 1 \
  --patch-size 8,32,32 --patches-per-volume 1 \
  --random-rotation-degrees 0 --num_workers 0 --log-interval 1
```

### Tune foreground imbalance

Draw every patch around a marker and increase its loss contribution:

```bash
python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
  --dataset-path data/NUMORPH_SEM_SEG_DATASET \
  --foreground-patch-probability 1.0 --class-weights 0.1,1.0 \
  --patches-per-volume 12 --lr 0.0001
```

Change one variable at a time and compare the held-out nucleus IoU using the same explicit `--validation-fold` and seeds.

## Dataset format and splitting

`--dataset-path` may be an extracted directory or ZIP containing BioImage Archive-style tables. `images.tsv` requires `image_id` and `filename`; `annotations.tsv` requires `image_id` and `filename` (an `annotation_id` is recommended):

```tsv
# images.tsv
image_id	filename
sample-001	images/sample-001.ome.tiff
```

```tsv
# annotations.tsv
annotation_id	image_id	filename
mask-001	sample-001	annotations/sample-001.ome.tiff
```

Paths are relative to the dataset root and every image must have exactly one mask. Images may be `YX`, `ZYX`, or `CZYX`; masks must contain only integer class values `0` and `1`. Non-singleton scene/time dimensions must be exported as separate records. Common BIA aliases such as `image_uuid`, `source_image_id`, `file_name`, and `file_path` are also accepted.

All records participate in a shuffled, group-stratified split. The held-out fold supplies both validation data during fitting and final test metrics; a run fits **one model**, not one model per fold. The resolved fold is printed and logged to MLflow as `selected_validation_fold`.

The [physical voxel-size resampling](#resampling-to-a-common-physical-voxel-size) is applied before patch selection. Training then adds random crops and foreground-aware sampling. Validation/test apply deterministic overlapping sliding-window inference to every voxel of each complete resampled volume, with no random augmentation; smaller volumes are padded and predictions are cropped back to their original shape. Raw logits are uniformly averaged in overlaps before argmax.

IoU and voxel accuracy are calculated once per epoch from globally reduced confusion counts rather than averages of batch scores. A class absent from both prediction and target has zero union and is assigned IoU 0; mean IoU always averages all configured classes.

## Other ways to run

### Conda

```bash
conda env create -f environment.yml
conda activate numorph-nuclei-segmentation
python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
  --dataset-path data/NUMORPH_SEM_SEG_DATASET \
  --max_epochs 100 --accelerator cpu --devices 1
```

Local runs write output below `lightning_logs/`.

### MLflow

The `MLproject` uses the published container and mounts `data/`, `mlruns/`, and `output/`:

```bash
python -m pip install mlflow==2.16.2
mlflow run . --build-image \
  -P dataset-path=/data -P max_epochs=100 \
  -P accelerator=cpu -P devices=1
```

Use `-A runtime=nvidia -P accelerator=gpu` for a GPU. MLflow parameters use `-P name=value`, while direct Python/Docker execution uses the corresponding `--name value` CLI flag.

## Export metric plots

Export loss, accuracy and IoU charts from the newest TensorBoard run as PNG (default) or SVG:

```bash
python tools/export_training_plots.py \
  --logdir mlruns --output-dir output/plots --format png
```

With Docker, mount both directories and run the same tool at `/app/tools/export_training_plots.py`. It is safe to rerun while training; metrics not yet available are skipped. In addition to loss, accuracy, and IoU, runs using the learning-rate monitor export `learning_rate.png` (or `.svg`), including separate series for multiple optimizer parameter groups.

## Outputs and compatibility

| Output | Docker location |
| --- | --- |
| Inference state dictionary | `/mlruns/numorph_unet3d.pt` |
| Best Lightning checkpoint | `/mlruns/checkpoints/` |
| TensorBoard events | `/mlruns/lightning_logs/` |

The state dictionary contains the plain `UNet3D` weights expected by `nuxnet-inference`; use its `--classes 2` option. Dependencies are pinned to Python 3.12, PyTorch 2.5.1, NumPy 1.26.4 and tifffile 2024.8.30. This project is released under the MIT license.

## FAIR model packaging and transfer learning

The reusable commands live in [`nidavellir_tools/`](nidavellir_tools/):

| Command | Responsibility |
| --- | --- |
| `build_model_package.py` | Build a new FAIR package from a declarative BioImage.IO RDF specification, a PyTorch checkpoint, test tensors, a model card, and optional provenance. |
| `model_package_registry.py` | Stage, verify, inspect, load, derive, validate, and publish an existing package. |

The tools are application-independent; scientific metadata and tensor semantics
come from the project's RDF specification and model card. Start from the
[annotated example specification](nidavellir_tools/examples/model-package.example.yaml),
then see the [FAIR packaging guide](docs/fair_model_packages.rst) for the package
contract, transfer-learning workflow, security boundaries, and publication
checklist. The root [`model-package.yaml`](model-package.yaml) is the concrete
NuxNet profile and must be reviewed for each released run.

### Package quick start

From the repository root, prepare a checkpoint, model card, and representative
input/output `.npy` fixtures, then run:

```bash
python nidavellir_tools/build_model_package.py \
  --specification model-package.yaml \
  --checkpoint lightning_logs/checkpoints/best.ckpt \
  --state-dict-key state_dict --strip-prefix model. \
  --test-input validation/test-input.npy \
  --test-output validation/test-output.npy \
  --model-card validation/README.md \
  --trace-input validation/model-ready-input.npy \
  --extra-file LICENSE \
  --output-dir output/model

python nidavellir_tools/model_package_registry.py inspect output/model
(cd output/model && sha256sum --check SHA256SUMS)
```

Stage the resulting archive and smoke-test a loading path:

```bash
python nidavellir_tools/model_package_registry.py stage \
  output/model.zip .model-cache/local-model

python nidavellir_tools/model_package_registry.py load \
  .model-cache/local-model --representation pytorch_state_dict \
  --weights-output work/parent.pt \
  --metadata-output work/parent.json

python nidavellir_tools/model_package_registry.py validate output/model.zip
```

For transfer learning, pass the paired weights and metadata produced by `load`:

```bash
mlflow run . \
  -P initial-weights=work/parent.pt \
  -P parent-metadata=work/parent.json \
  -P dataset-path=/new-data
```

Both options are mandatory together. Treat packaged Python architecture code as
executable code: inspect and trust it before loading. BioImage.IO validation is a
technical check, not evidence of scientific validity.

## Development and documentation

Install the runtime and test dependencies, then run the same test command as CI:

```bash
python -m pip install -r requirements.txt pytest==8.3.3
pytest -q
```

Build the Sphinx documentation locally with:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The repository intentionally has no configured Python linter. CI runs the test
suite and container smoke tests; formatting is governed only by `.editorconfig`.
When changing a CLI option, update `MLproject`, this README, and the relevant
tests together.
