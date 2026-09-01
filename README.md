<!-- markdownlint-disable MD010 MD013 -->

# numorph-nuclei-segmentation

Reproducible 3D U-Net training for nucleus-marker segmentation in the NuMorph light-sheet microscopy dataset. The training architecture and exported state dictionary are compatible with `nuxnet-inference`.

- Free software: MIT
- Input data: BIA/BioImage.IO metadata with paired OME-TIFF volumes
- Task: binary nucleus-marker detection

## Dataset overview

`NUMORPH_SEM_SEG_DATASET` is a curated training-data package derived from the NuMorph 3D U-Net nucleus-detection data described by Krupa _et al._ in [_NuMorph: Tools for cortical cellular phenotyping in tissue-cleared whole-brain images_](https://doi.org/10.1016/j.celrep.2021.109802). It contains light-sheet fluorescence microscopy patches of TO-PRO-3-labelled nuclei from wild-type mouse (`NCBITaxon:10090`) cerebral cortex (`UBERON:0000956`).

The public archive contains 32 paired image/mask volumes split evenly between two physical-resolution groups:

| Group  | Pairs | Voxel size (X × Y × Z) | Volume shapes (Z × Y × X)                   |
| ------ | ----: | ---------------------- | ------------------------------------------- |
| `C075` |    16 | 0.75 × 0.75 × 2.5 µm   | 16 × `64 × 224 × 224`                       |
| `C121` |    16 | 1.21 × 1.21 × 4.0 µm   | 14 × `64 × 224 × 224`; 2 × `64 × 256 × 256` |

Images are normalized, single-channel `float32` OME-TIFF volumes with values in `[0,1]`. Their paired masks are `uint8` OME-TIFF volumes with binary values `{0,1}`. The foreground occupies roughly 1–5 percent of a volume in the current package.

> [!IMPORTANT]
> The masks represent manually curated **2D middle-Z nucleus markers for detection**. They are not complete 3D nuclear boundaries or nucleus instance segmentations, even though they are stored in 3D OME-TIFF volumes.

The archive also includes BIA `images.tsv`/`annotations.tsv` file lists, REMBI/MIFA submission notes, a BioImage.IO dataset RDF and manifest, and a machine-readable dataset summary. Its metadata currently records the license, BioImage Archive accession, and BioImage.IO resource ID as pending; users should therefore confirm reuse terms before redistributing the data.

## Architecture

The model is the same compact 2.34-million-parameter 3D U-Net used by `nuxnet-inference`. Its encoder has 32-, 64-, and 128-channel levels, two learned stride-2 downsampling operations, and a 128-channel bottleneck. The decoder uses nearest-neighbour 2x upsampling and concatenates the corresponding encoder features through U-Net skip connections before producing per-voxel logits with a 1x1x1 convolution. Each convolutional block contains two 3x3x3 convolutions with 3D dropout, batch normalization, and ReLU activation. The default output has two channels: background and nucleus marker.

This detection task sits in the scientific context of [Krupa et al.'s NuMorph workflow](https://doi.org/10.1016/j.celrep.2021.109802), an open-source analysis toolkit developed to register, segment, and quantify cellular distributions in cleared whole-mouse-brain microscopy data. NuMorph combines brain registration and cortical analysis with deep-learning-based cell detection so detected cells can be mapped into anatomical space and summarized across regions. This repository trains the nucleus-marker network used by that detection stage; it does **not** reconstruct full nuclear surfaces or instances. The encoder-decoder design follows the localization principle of [U-Net](https://doi.org/10.1007/978-3-319-24574-4_28), extended here to volumetric convolutions and anisotropic 3D image patches.

### Training objective and optimization

The network's logits are converted to class probabilities with softmax and optimized with the [focal loss of Lin et al.](https://doi.org/10.1109/TPAMI.2018.2858826). For the target-class probability $p_t$, the implemented objective is $-\alpha_t(1-p_t)^\gamma\log(p_t)$, with fixed $\gamma=2$ and slight label smoothing (`1e-5`). The focusing term reduces the contribution of already easy voxels, while `--class-weights` controls the relative class contribution. The defaults `0.2,1.0` are normalized internally and emphasize the sparse nucleus-marker class over the much more abundant background; supply exactly one comma-separated weight per output class.

Training uses Adam (`--lr 0.0001` by default) and reduces the learning rate by a factor of ten after ten epochs without improvement in epoch-level training loss, down to `1e-6`. Checkpoint selection is separate: Lightning retains the model with the lowest validation loss. Runs report loss, voxel accuracy, per-class intersection-over-union (IoU), and mean IoU for training, validation, and test phases. Accuracy can be dominated by background in this sparse task, so nucleus-class IoU and mean IoU should be considered alongside it.

## Run a training job

### Docker (recommended)

Docker provides the pinned Python and CUDA dependencies, so no local Python environment is
required. You need Docker Engine and an extracted dataset directory or dataset ZIP. The image is
CUDA-enabled, but it can run in CPU mode on a host without a GPU.

From the repository root, create host directories for persistent output and build the image:

```bash
mkdir -p "$PWD/mlruns" "$PWD/output"
docker build -t numorph-nuclei-segmentation .
```

Mount the dataset read-only at `/data` and run a CPU training job:

```bash
docker run --rm \
  -v "$PWD/dataset:/data:ro" \
  -v "$PWD/mlruns:/mlruns" \
  -v "$PWD/output:/output" \
  numorph-nuclei-segmentation \
  --dataset-path /data --max_epochs 100 --accelerator cpu --devices 1
```

`$PWD/dataset` may be an extracted dataset directory. To use a ZIP instead, mount the file and
point `--dataset-path` to its in-container path:

```bash
docker run --rm \
  -v "$PWD/data/NUMORPH_SEM_SEG_DATASET.zip:/data/dataset.zip:ro" \
  -v "$PWD/mlruns:/mlruns" \
  numorph-nuclei-segmentation \
  --dataset-path /data/dataset.zip --max_epochs 100 --accelerator cpu --devices 1
```

For NVIDIA GPU training, first install the NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. Then expose the GPU to the container and select Lightning's GPU accelerator:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/data:ro" \
  -v "$PWD/mlruns:/mlruns" \
  -v "$PWD/output:/output" \
  numorph-nuclei-segmentation \
  --dataset-path /data --max_epochs 100 --accelerator gpu --devices 1
```

The bind mounts have distinct purposes:

| Container path | Purpose | Required |
| -------------- | ------- | -------- |
| `/data` | Input dataset directory or ZIP | Yes |
| `/mlruns` | Lightning logs, MLflow artifacts, and `numorph_unet3d.pt` | Recommended |
| `/output` | Optional destination for exported metric plots | No |

Without the `/mlruns` bind mount, training results are deleted with an `--rm` container. The
container runs the training module automatically when its arguments begin with an option. It can
also run an explicit command, which is useful for inspecting help or opening a shell:

```bash
docker run --rm numorph-nuclei-segmentation --help
docker run --rm -it --entrypoint /bin/bash numorph-nuclei-segmentation
```

All options shown by `--help` can be appended to the Docker command. A completed run writes the
best inference-compatible weights to `/mlruns/numorph_unet3d.pt`.

### Publish a test container

Repository maintainers can build a disposable personal image without creating a release. In the
GitHub repository, open **Actions**, select **Publish personal test container to GHCR**, and choose
**Run workflow**. The workflow publishes the current revision for `linux/amd64` as both
`ghcr.io/luiskuhn/numorph-nuclei-segmentation-test:latest` and `:test` using the repository's
automatic `GITHUB_TOKEN`; no personal access-token secret is required. Make the package public in
its GHCR package settings if it should be pullable without authentication.

Pushes to `main` publish the regular repository-owned image with `main` and `latest` tags. A
published semver release additionally publishes full-version and major/minor tags.

### MLflow with Docker

The repository's `MLproject` uses the published container image by default and mounts the local
`data/`, `mlruns/`, and `output/` directories. Put an extracted dataset (or its ZIP) below `data/`,
install MLflow, and run:

```bash
python -m pip install mlflow==2.16.2
mlflow run . --build-image \
  -P dataset-path=/data -P max_epochs=100 \
  -P accelerator=cpu -P devices=1
```

For one GPU, expose it to Docker and change the Lightning accelerator:

```bash
mlflow run . --build-image -A runtime=nvidia \
  -P dataset-path=/data -P max_epochs=100 \
  -P accelerator=gpu -P devices=1
```

Both commands use five folds and select the held-out validation/test fold reproducibly from
`general-seed=0`. Add `-P validation-fold=3` to hold out fold 3 explicitly, or change the fold count
with `-P cross-validation-folds=10`. `--build-image` builds from the local `Dockerfile`; omit it to
use the image configured in `MLproject`. MLflow writes run metadata and artifacts below `mlruns/`.

### Conda (without Docker or MLflow)

To run directly on the host, create the pinned Conda environment and invoke the module:

```bash
conda env create -f environment.yml
conda activate numorph-nuclei-segmentation
python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
  --dataset-path data/NUMORPH_SEM_SEG_DATASET --max_epochs 100 \
  --accelerator cpu --devices 1 --cross-validation-folds 5
```

Use `python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation --help` for the complete
CLI. A completed local run writes the best inference-compatible weights to
`lightning_logs/numorph_unet3d.pt`.

## OME-TIFF / BioImage Archive dataset format

Training data are read directly from OME-TIFF files and described by two BioImage Archive-style, tab-separated metadata tables in `--dataset-path`. Paths must be relative to that directory and each image must have exactly one segmentation mask. Multi-channel images are returned as `CZYX` tensors; labels are single-channel integer `ZYX` tensors. Every mask is validated when loaded and must contain only the discrete voxel values `0` (background) and `1` (nucleus marker); any fractional, negative, or other class value is rejected. Two-dimensional `YX` data are accepted as one-slice volumes. Non-singleton scene or time dimensions must be exported as separate image records before training.

`images.tsv` requires `image_id` and `filename`. A legacy `split` column may be present, but cross-validation assigns folds across all records:

```tsv
image_id	filename	split
sample-001	images/sample-001.ome.tiff	train
sample-002	images/sample-002.ome.tiff	test
```

`annotations.tsv` requires `image_id` and `filename`; `annotation_id` is recommended for submission provenance:

```tsv
annotation_id	image_id	filename
mask-001	sample-001	annotations/sample-001.ome.tiff
mask-002	sample-002	annotations/sample-002.ome.tiff
```

The loader also understands the BIA export aliases `image_uuid`, `source_image_id`, `source_image_uuid`, `file_name`, and `file_path`.

## Cross-validation

Every training run uses a shuffled, group-stratified cross-validation split. The default is five folds (`--cross-validation-folds 5`). Records within each `resolution_group`/`subset` are shuffled using `--general-seed` and distributed across folds, keeping acquisition groups represented as evenly as the data permit. One fold supplies both validation metrics during fitting and final test metrics; the remaining folds supply training data. This is fold-based holdout evaluation for a single run, not five consecutive model fits.

By default (`--validation-fold 0`), a fold is selected pseudo-randomly and reproducibly from `--general-seed`. To select one explicitly, pass its one-based number, for example `--validation-fold 3`. The resolved fold is printed as `Cross-validation fold: 3/5` and recorded in MLflow as `selected_validation_fold`, so every run documents which data measured its validation/test metrics. The number of folds must be at least two and cannot exceed the number of image/mask pairs.

The Conda environment, pip requirements, CI, and container use Python 3.12, PyTorch 2.5.1, NumPy
1.26.4, and tifffile 2024.8.30 to match nuxnet-inference. The container is based on NVIDIA CUDA
12.4.1 with cuDNN on Ubuntu 22.04, includes a dependency-import health check, and uses Miniforge to
provide Python 3.12.

### Export TensorBoard metrics as PNG files

`tools/export_training_plots.py` finds the newest TensorBoard run below a log directory and
exports every available train/validation/test metric group as a single plot. It can be rerun
while training is active; metrics not logged yet are skipped and existing PNGs are replaced
atomically. From a Python environment containing the project requirements, run:

```bash
python tools/export_training_plots.py \
  --logdir /mlruns \
  --output-dir /mlruns/plots
```

The image already contains TensorBoard and Matplotlib, and `/app/tools/export_training_plots.py`
is copied into it during the image build. To start a named training container with persistent
training logs and plot output, run:

```bash
mkdir -p "$PWD/mlruns" "$PWD/output"
docker build -t numorph-nuclei-segmentation .
docker run --name nuxnet-fold1 --rm --gpus all \
  -v "$PWD/dataset:/data" \
  -v "$PWD/mlruns:/mlruns" \
  -v "$PWD/output:/output" \
  numorph-nuclei-segmentation \
  --dataset-path /data --max_epochs 100 --accelerator gpu --devices 1
```

In a second terminal, export from the running `nuxnet-fold1` container without stopping
training:

```bash
docker exec nuxnet-fold1 \
  python /app/tools/export_training_plots.py \
  --logdir /mlruns \
  --output-dir /output/plots
```

The output path must be mounted if the plots should persist on the host. For example, start
the container with `-v "$PWD/output:/output"`; the command above then creates the PNGs in
`$PWD/output/plots` on the Docker host. Alternatively, when the `/mlruns` bind mount is used, choose
`--output-dir /mlruns/plots`; the files appear in the host directory mounted at `/mlruns`
(for the example Docker command above, `$PWD/mlruns/plots`). The exporter automatically uses
the newest run below `/mlruns`, and it is safe to execute the command repeatedly as new events
are logged.

For a one-off export when no training container is currently running, mount existing logs and
an output directory into a temporary container:

```bash
docker run --rm \
  -v "$PWD/mlruns:/mlruns:ro" \
  -v "$PWD/output:/output" \
  numorph-nuclei-segmentation \
  python /app/tools/export_training_plots.py \
    --logdir /mlruns --output-dir /output/plots
```

To copy plots from a remote training machine to a local Mac, run this on the Mac (replace the
host and path with the bind-mounted host path):

```bash
mkdir -p ~/Downloads/nuxnet-plots
scp 'user@training-host:/path/to/nuxnet-training/mlruns/plots/*.png' \
  ~/Downloads/nuxnet-plots/
```

## Dataset download and training behavior

The dataset can be fetched directly from the [public Google Drive share](https://drive.google.com/file/d/1nwLPXoWEsBb3wLNwXHY3L23UiO-5IIyn/view?usp=drive_link). The ZIP is approximately 154 MiB and its OME-TIFF data expand to approximately 500 MiB. The download is written atomically to `--dataset-path` and an existing file is reused, making repeated training runs inexpensive:

```bash
python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
  --download-dataset --dataset-path data/numorph_nuclei_seg_ds.zip
```

The configured default is the NuMorph Google Drive file. To download a dataset from another public HTTP(S) link, add `--dataset-url URL`. Use `--overwrite-dataset` to replace an existing archive. The link must allow unauthenticated downloads; private Google Drive files are not supported.

`--dataset-path` accepts either the extracted dataset directory or the ZIP archive itself. The loader searches through a BioImage.IO wrapper directory for the single BIA `images.tsv`/`annotations.tsv` pair, so the archive does not need to be rearranged before training. In addition to the canonical column names above, columns named `image`, `source_image`, `file`, `filepath`, `uri`, `annotation`, `mask`, or `label` are supported. Header matching is insensitive to spaces, hyphens, and case. Annotation source references may use the image identifier, the relative image path, or the image basename. If `images.tsv` has no explicit identifier column, the image path is used as its stable identifier.

The network is source-compatible with `nuxnet-inference`'s `UNet3D`: it has the same block structure, channel widths, layer names, and logits output. Training uses normalized `32,128,128` (Z,Y,X) patches, corresponding to 128x128x32-voxel chunks, which makes differently sized OME-TIFF volumes batchable and guarantees the spatial dimensions required by the model's two downsampling stages. Training chunks are sampled randomly and rotated by up to 2 degrees about a random 3D axis. Image intensities are interpolated trilinearly while masks use nearest-neighbour interpolation, ensuring augmented masks remain discrete `int64` tensors containing only `0` and `1`. Validation and test patches use centered crops without augmentation. Volumes smaller than the requested patch are zero padded. Set a dataset-sized patch with `--patch-size Z,Y,X`; all three values must be multiples of four, and control or disable rotation with `--random-rotation-degrees` (use `0` to disable it). Disable the inference-compatible min/max normalization with `--no-normalize-input` only when the data have already been normalized.

After training, `numorph_unet3d.pt` contains the plain `UNet3D.state_dict` expected by the inference CLI (rather than Lightning's wrapper keys), and is also recorded as an MLflow model artifact.

## NuMorph dataset defaults

The defaults are tailored to the `NUMORPH_SEM_SEG_DATASET` package. Its 32 pairs are discovered from `bia/images.tsv` and `bia/annotations.tsv`; `Files` entries are resolved relative to the package root. `Resolution group` and `Pair ID` form unique sample IDs such as `C075:0001`. Cross-validation distributes the 16 C075 and 16 C121 volumes across folds so both physical-resolution groups remain represented.

The image and mask properties, dimensions, and annotation semantics are summarized in [Dataset overview](#dataset-overview). Training defaults to one input channel and two output classes. To account for the sparse nucleus markers, each training volume produces eight patches per epoch, 80 percent of patches are centered on a foreground marker, and focal-loss weights default to `0.2,1.0` for background and nucleus-marker voxels. These can be changed with `--patches-per-volume`, `--foreground-patch-probability`, and `--class-weights`.

The annotations are nucleus-detection marker masks, not complete 3D nucleus instance segmentations. Use the exported checkpoint with the inference tool's `--classes 2` option.
