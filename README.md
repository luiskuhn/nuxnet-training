<!-- markdownlint-disable MD013 -->

# nuxnet-training

![Graphical abstract of the nuxnet-training workflow](docs/images/graph_abstract_nuxnet_training.png)

Reproducible 3D U-Net training for nucleus-marker detection in NuMorph light-sheet microscopy data. The exported PyTorch state dictionary is compatible with [`nuxnet-inference`](https://github.com/luiskuhn/nuxnet-inference).

## What this repository trains

The model is a compact residual 3D U-Net with 32-, 64-, and 128-channel feature levels. Two stride-2 convolutions form the encoder, and two nearest-neighbor upsampling stages restore the original resolution while reusing encoder features through skip connections. Every two-convolution block has a residual shortcut (a normalized 1×1×1 projection when channels change), and the final 1×1×1 head emits two-channel background/marker logits. Spatial dropout acts only on learned feature maps, never directly on the raw input.

Training defaults to unweighted cross-entropy plus foreground soft Dice (focal loss with fixed $\gamma=2$ remains selectable), Adam, validation-loss checkpointing and reports loss, voxel accuracy, per-class IoU and mean IoU. Because the foreground is sparse, prefer nucleus-class and mean IoU over accuracy when comparing runs. The architecture remains below 2.5 million parameters and retains the compact model's memory profile.

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

The U-Net head returns raw two-channel logits. New runs default to `--loss-function dice-ce`, which computes

```text
loss = ce_loss_weight × cross_entropy(logits, target)
     + dice_loss_weight × foreground_soft_dice_loss(logits, target)
```

Cross-entropy receives the raw logits. Foreground Dice applies softmax internally, selects class 1, and aggregates intersection and denominator over the complete batch and all spatial dimensions with smoothing for finite empty-foreground behavior. Equal `--class-weights 1.0,1.0` means ordinary unweighted cross-entropy; non-uniform weights are an explicit experiment. The prior focal objective remains available through `--loss-function focal`.

Metrics and packaged BioImage.IO inference apply softmax explicitly when probabilities are needed. Plain `UNet3D` state-dictionary and TorchScript consumers receive logits and must apply `softmax(dim=1)` themselves. They must also resample source OME data to the target Z,Y,X grid before tensor inference; the exported provenance and `rdf.yaml` record that grid but the tensor model cannot infer source voxel calibration from a tensor alone.

## Quick start with Docker

Docker is recommended because it provides pinned Python and CUDA dependencies. Build the image, create persistent output directories, and train on an extracted dataset:

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
| LR reduction factor | `--lr-scheduler-factor` | `0.5` | Multiply the learning rate by this factor when training loss plateaus. |
| LR scheduler patience | `--lr-scheduler-patience` | `5` | Number of stagnant training-loss epochs tolerated before reducing the rate. |
| LR improvement threshold | `--lr-scheduler-threshold` | `1e-5` | Minimum absolute training-loss improvement that resets scheduler patience. |
| LR scheduler cooldown | `--lr-scheduler-cooldown` | `2` | Epochs to wait after a rate reduction before resuming plateau checks. |
| Minimum learning rate | `--min-lr` | `1e-6` | Lower bound for learning-rate reductions. |
| Patch size | `--patch-size Z,Y,X` | `32,128,128` | Spatial context per sample. Each dimension must be divisible by four; larger patches require more memory. |
| Target physical voxel size | `--target-voxel-spacing Z,Y,X` | `3.0,1.0,1.0` | OME-style physical size of one output voxel in Z,Y,X order, in µm per voxel. |
| Training batch size | `--training-batch-size` | `1` | Patches per optimizer step. Increase only when memory permits. |
| Patches per volume | `--patches-per-volume` | `8` | Random training patches drawn from each volume per epoch. Higher values provide more sampling at greater runtime. |
| Foreground sampling | `--foreground-patch-probability` | `0.8` | Probability that a training patch is centered on a marker (`0`–`1`). |
| Loss | `--loss-function` | `dice-ce` | Combined foreground Dice plus cross-entropy, or legacy `focal`. |
| Loss weights | `--ce-loss-weight`, `--dice-loss-weight` | `1.0`, `1.0` | Contributions to the combined Dice+CE objective. |
| Class weights | `--class-weights` | `1.0,1.0` | Optional explicit per-class weights; equal values give unweighted cross-entropy. |
| Dropout | `--dropout-rate` | `0.10` | Probability for all 3D dropout layers: two per residual convolution block and one after each down/up transition. |
| Rotation | `--random-rotation-degrees` | `10.0` | Maximum continuous XY training rotation; Z is never mixed with lateral axes. |
| Right-angle rotation | `--random-rotation-90-probability` | `0.5` | Probability of an exact random 0°, 90°, 180°, or 270° XY rotation. |
| Folds | `--cross-validation-folds` | `5` | Number of shuffled, resolution-group-stratified folds (at least two). |
| Held-out fold | `--validation-fold` | `0` | One-based validation/test fold. `0` selects reproducibly from the general seed. |
| Validation interval | `--test-epochs` | `10` | Run validation every N epochs. Final testing always uses the best checkpoint. |
| Input normalization | `--normalize-input` / `--no-normalize-input` | enabled | Per-volume min/max normalization compatible with inference. Disable only for prepared inputs. |

Reproducibility and execution options:

| CLI flag | Default | Description |
| --- | ---: | --- |
| `--general-seed`, `--pytorch-seed` | `0`, `0` | Python/NumPy and PyTorch random seeds. |
| `--accelerator` | `auto` | Lightning accelerator: `auto`, `cpu`, or `gpu`. |
| `--devices` | `auto` | Device count or `auto`; use with `--strategy` for distributed training. |
| `--num_workers` | `2` | Data-loader workers. Increase when input loading limits throughput. |
| `--log-interval` | `100` | Training steps between log writes. |
| `--n-channels`, `--n-class` | `1`, `2` | Input channels and output classes. Defaults match NuMorph and inference. |
| `--test-batch-size` | `1` | Validation/test batch size. |
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
  --loss-function dice-ce --ce-loss-weight 1.0 --dice-loss-weight 1.0 \
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

The [physical voxel-size resampling](#resampling-to-a-common-physical-voxel-size) is applied before patch selection. Training then adds random crops and foreground-aware sampling; validation/test use centered crops and no random augmentation. Smaller normalized volumes are zero padded.

## Other ways to run

### Conda

```bash
conda env create -f environment.yml
conda activate nuxnet-training
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

## FAIR model export

`tools/export_model.py` turns one trained checkpoint into two deliverables:

1. an unpacked model repository at `--output-dir`, intended for review and upload to Hugging Face; and
2. a sibling ZIP archive, intended for validation and upload to BioImage.IO.

The tool requires the project's Python environment because it loads the checkpoint and performs a real CPU inference and TorchScript trace. Create the pinned Conda environment first, or use the project image:

```bash
conda env create -f environment.yml
conda activate nuxnet-training
python tools/export_model.py --help
```

The `--checkpoint` input may be either the plain `numorph_unet3d.pt` written at the end of training or the best `.ckpt` file under `checkpoints/`. The exporter recognizes Lightning's `state_dict`, removes its `model.` prefix, and strictly loads all tensors into `UNet3D`; incompatible channels, classes, dropout-independent architecture keys, missing tensors, or unexpected tensors stop the export.

### Export a model

Supply publication-specific metadata rather than copying the placeholders below:

```bash
python tools/export_model.py \
  --checkpoint mlruns/numorph_unet3d.pt \
  --output-dir output/nuxnet-model \
  --name "NuMorph nucleus-marker U-Net" \
  --description "3D U-Net for nucleus-marker detection in cleared mouse-brain microscopy" \
  --author "Your Name" --author-orcid 0000-0000-0000-0000 \
  --github-user YOUR_GITHUB_USERNAME --model-version 1.0.0 \
  --citation "Authors (year), title" --doi 10.example/article \
  --dataset "persistent dataset identifier" --dataset-version "1" \
  --mlflow-run-id YOUR_RUN_ID
```

`--doi` and `--citation-url` are alternatives; at least one is required so the citation is resolvable. `--source-repository` and `--git-commit` override the current Git checkout when exporting a model trained elsewhere. `--cover` replaces the default graphical abstract with a representative GIF, JPEG, PNG, or SVG. `--test-shape Z,Y,X` controls the small technical test volume and defaults to `4,16,16`; all dimensions must be at least four and divisible by four.

The architecture options must match the training run: `--input-channels 1`, `--classes 2`, and `--dropout 0.10` are the defaults. Input normalization is enabled by default, matching normal NuxNet training. Pass `--no-normalize-input` only if that checkpoint was trained with `--no-normalize-input`; the choice is reflected in the test tensors, RDF preprocessing, model card, and provenance. Existing non-empty output directories are protected unless `--overwrite` is supplied.

The equivalent Docker invocation is:

```bash
docker run --rm \
  -v "$PWD/mlruns:/mlruns:ro" -v "$PWD/output:/output" \
  --entrypoint python nuxnet-training \
  /app/tools/export_model.py \
  --checkpoint /mlruns/numorph_unet3d.pt \
  --output-dir /output/nuxnet-model \
  --name "NuMorph nucleus-marker U-Net" \
  --description "3D U-Net for nucleus-marker detection" \
  --author "Your Name" --github-user YOUR_GITHUB_USERNAME \
  --citation "Authors (year), title" --doi 10.example/article \
  --dataset "persistent dataset identifier" --dataset-version "1"
```

During export, the tool loads and strictly checks the weights, creates deterministic raw test data, applies the configured normalization, runs the model, computes expected post-softmax probabilities, traces TorchScript, writes metadata and checksums, and finally creates the ZIP. `test-input.npy` and `test-output.npy` are technical reproducibility fixtures; they are not evaluation data and do not establish scientific model quality.

### Exactly what is produced

| File | BioImage.IO role | Hugging Face role |
| --- | --- | --- |
| `rdf.yaml` | Required 0.5.14 resource description: identity, authors/maintainer, citation, cover, tensor axes, preprocessing, postprocessing, test tensors, weights, dependencies, and hashes. | Supplementary machine-readable interoperability metadata. |
| `model.ts` | Portable TorchScript weights referenced by the RDF; the network returns logits. | Ready-to-download TorchScript representation. |
| `weights.pt` | Alternative tensor-only PyTorch state dictionary referenced by the RDF. | Native PyTorch weights for downstream loading and fine-tuning. |
| `unet_3d_models.py` | Exact callable architecture needed to reconstruct `weights.pt`. | Human-readable/loading implementation. |
| `test-input.npy` | Deterministic raw `BCZYX` input used by `bioimageio test`. | Reproducibility fixture. |
| `test-output.npy` | Expected probability tensor after RDF min/max preprocessing, model execution, and softmax postprocessing. | Reproducibility fixture. |
| `cover.png` | Required representative cover; defaults to the project graphical abstract and may be replaced with `--cover`. | Visual repository asset. |
| `README.md` | BioImage.IO model documentation with a Validation section. | Hugging Face model card with Hub YAML front matter. |
| `environment.yml` | Pinned state-dictionary dependencies. | Reproducible Conda environment. |
| `provenance.json` | Additional FAIR provenance: model/checkpoint digests, dataset version, MLflow run, software, repository, commit, and export command. | Machine-readable provenance. |
| `LICENSE` | Distribution terms. | Distribution terms. |
| `ARTIFACTS.md` | Inventory and upload mapping included inside the package. | Inventory and upload mapping. |
| `SHA256SUMS` | Integrity manifest for all package files. | Integrity manifest. |

For the example command, the resulting layout is:

```text
output/
├── nuxnet-model.zip
└── nuxnet-model/
    ├── ARTIFACTS.md
    ├── LICENSE
    ├── README.md
    ├── SHA256SUMS
    ├── cover.png
    ├── environment.yml
    ├── model.ts
    ├── provenance.json
    ├── rdf.yaml
    ├── test-input.npy
    ├── test-output.npy
    ├── unet_3d_models.py
    └── weights.pt
```

The ZIP contains the directory files at its archive root and is the unit uploaded to BioImage.IO. For Hugging Face, upload the individual directory contents to the root of a model repository; the Hub renders `README.md` as the model card and exposes both weight formats. The RDF declares per-volume min/max scaling across `Z/Y/X` when enabled and softmax over the output channel, so BioImage.IO consumers receive background/marker probabilities while direct TorchScript and PyTorch users receive raw logits.

### Consume the exported weights directly

Load the native state dictionary for Python inference:

```python
import torch
from numorph_nuclei_segmentation.model import UNet3D

model = UNet3D(in_channels=1, classes=2, dropout=0.10)
model.load_state_dict(torch.load("output/nuxnet-model/weights.pt", weights_only=True))
model.eval()
```

Or load the self-contained TorchScript graph without importing the architecture:

```python
import torch

model = torch.jit.load("output/nuxnet-model/model.ts", map_location="cpu")
model.eval()
```

Both direct representations return logits and expect preprocessing consistent with the model card. Apply `softmax(dim=1)` when probabilities are required. BioImage.IO performs the RDF-declared normalization and softmax automatically.

### Validate and publish

FAIR metadata is only useful when it is specific: replace every example value, use persistent ORCID/DOI/dataset identifiers where available, record the MLflow run, and add the run's quantitative validation results to the generated `README.md`. Dataset licensing is independent of the model license; no training data are redistributed. In accordance with the [BioImage.IO developer guide](https://bioimage.io/docs/#/guides/developers-guide?id=models-in-the-bioimage-model-zoo), the package includes weights, example input/output arrays, an RDF, a representative cover, documentation, explicit preprocessing/postprocessing, and every local execution dependency. Validate the actual archive in a current BioImage.IO environment before upload:

```bash
conda create --name bioimageio -c conda-forge bioimageio.core pytorch
conda activate bioimageio
bioimageio test output/nuxnet-model.zip
```

Alternatively, if `bioimageio.core` and its `bioimageio` executable are already installed in the export environment, append `--validate` to the export command to run that check immediately after packaging. A successful technical validation does not replace reporting held-out IoU and other scientific evaluation results in the generated model card.

After validation:

* **BioImage.IO:** sign in at [bioimage.io](https://bioimage.io), choose **Upload**, submit `output/nuxnet-model.zip`, review the parsed metadata, validate, and address any reviewer feedback.
* **Hugging Face:** create a model repository and upload everything inside `output/nuxnet-model/` to its root. Do not upload only `weights.pt`: the model card, provenance, license, architecture, environment, tests, and checksums are the reusable FAIR record.
