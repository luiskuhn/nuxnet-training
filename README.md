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

The model is the same reduced 3D U-Net used by `nuxnet-inference`. It has two downsampling stages, nearest-neighbor decoder upsampling, skip connections, 3D convolutions, batch normalization, dropout, and a two-channel logits output for background and nucleus markers.

## OME-TIFF / BioImage Archive dataset format

Training data are read directly from OME-TIFF files and described by two BioImage Archive-style, tab-separated metadata tables in `--dataset-path`. Paths must be relative to that directory and each image must have exactly one segmentation mask. Multi-channel images are returned as `CZYX` tensors; labels are single-channel integer `ZYX` tensors. Two-dimensional `YX` data are accepted as one-slice volumes. Non-singleton scene or time dimensions must be exported as separate image records before training.

`images.tsv` requires `image_id` and `filename`. It may contain a `split` column whose values are `train`, `validation`/`val`, or `test`:

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

The loader also understands the BIA export aliases `image_uuid`, `source_image_id`, `source_image_uuid`, `file_name`, and `file_path`. If no split column is supplied, a deterministic split is generated using `--test-percent` and `--general-seed`.

The environment and container use Python 3.12, PyTorch 2.5.1, NumPy 1.26.4, and tifffile 2024.8.30 to match nuxnet-inference. Build and run locally with:

```bash
docker build -t numorph-nuclei-segmentation .
docker run --rm -v "$PWD/dataset:/data" -v "$PWD/mlruns:/mlruns" numorph-nuclei-segmentation \
  --dataset-path /data --max_epochs 100 --accelerator auto --devices auto
```

## Dataset download and training behavior

The dataset can be fetched directly from the [public Google Drive share](https://drive.google.com/file/d/1nwLPXoWEsBb3wLNwXHY3L23UiO-5IIyn/view?usp=drive_link). The ZIP is approximately 154 MiB and its OME-TIFF data expand to approximately 500 MiB. The download is written atomically to `--dataset-path` and an existing file is reused, making repeated training runs inexpensive:

```bash
python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation \
  --download-dataset --dataset-path data/numorph_nuclei_seg_ds.zip
```

The configured default is the NuMorph Google Drive file. To download a dataset from another public HTTP(S) link, add `--dataset-url URL`. Use `--overwrite-dataset` to replace an existing archive. The link must allow unauthenticated downloads; private Google Drive files are not supported.

`--dataset-path` accepts either the extracted dataset directory or the ZIP archive itself. The loader searches through a BioImage.IO wrapper directory for the single BIA `images.tsv`/`annotations.tsv` pair, so the archive does not need to be rearranged before training. In addition to the canonical column names above, columns named `image`, `source_image`, `file`, `filepath`, `uri`, `annotation`, `mask`, or `label` are supported. Header matching is insensitive to spaces, hyphens, and case. Annotation source references may use the image identifier, the relative image path, or the image basename. If `images.tsv` has no explicit identifier column, the image path is used as its stable identifier.

The network is source-compatible with `nuxnet-inference`'s `UNet3D`: it has the same block structure, channel widths, layer names, and logits output. Training uses normalized fixed-size patches (`16,64,64` by default), which makes differently sized OME-TIFF volumes batchable and guarantees the spatial dimensions required by the model's two downsampling stages. Training patches are sampled randomly; validation and test patches use centered crops. Volumes smaller than the requested patch are zero padded. Set a dataset-sized patch with `--patch-size Z,Y,X`; all three values must be multiples of four. Disable the inference-compatible min/max normalization with `--no-normalize-input` only when the data have already been normalized.

After training, `numorph_unet3d.pt` contains the plain `UNet3D.state_dict` expected by the inference CLI (rather than Lightning's wrapper keys), and is also recorded as an MLflow model artifact.

## NuMorph dataset defaults

The defaults are tailored to the `NUMORPH_SEM_SEG_DATASET` package. Its 32 pairs are discovered from `bia/images.tsv` and `bia/annotations.tsv`; `Files` entries are resolved relative to the package root. `Resolution group` and `Pair ID` form unique sample IDs such as `C075:0001`. The seeded train/test split is stratified across the 16 C075 and 16 C121 volumes so both physical-resolution groups remain represented.

The image and mask properties, dimensions, and annotation semantics are summarized in [Dataset overview](#dataset-overview). Training defaults to one input channel and two output classes. To account for the sparse nucleus markers, each training volume produces eight patches per epoch, 80 percent of patches are centered on a foreground marker, and focal-loss weights default to `0.2,1.0` for background and nucleus-marker voxels. These can be changed with `--patches-per-volume`, `--foreground-patch-probability`, and `--class-weights`.

The annotations are nucleus-detection marker masks, not complete 3D nucleus instance segmentations. Use the exported checkpoint with the inference tool's `--classes 2` option.
