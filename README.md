# DINOv3 Cellpose

Cellpose turns a cell image into two inward flow fields and one foreground
logit, then follows those flows to recover individual masks. This repository
keeps only that 2D path and uses a DINOv3 ViT encoder, with ViT-B/16 as the
default.

This project uses ordinary fine-tuning without LoRA. It provides a DINOv3
Cellpose segmentation baseline with optional instance classification. SAM,
the legacy U-Net, 3D inference, the GUI, restoration, model-zoo downloads, and
distributed training are not included.

Fine-tuning also supports optional per-instance class prediction: an image,
instance mask and JSON mapping of mask IDs to class names form one training
sample. Start with `config/classify.yaml`; see [label format and commands](docs/labels.md).
The default `config/train.yaml` remains segmentation-only.

## Installation

DINOv3 uses the separate
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
Review and accept it before installing the dependency or using its weights.
The pretrained weights require access through the
[official DINOv3 repository](https://github.com/facebookresearch/dinov3#pretrained-models).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Select a DINOv3 ViT backbone with `model.backbone` in
`config/train.yaml`. Supported values are `vits16`, `vits16plus`,
`vitb16`, `vitl16`, `vith16plus`, and `vit7b16`. The checkpoint in
`model.backbone_weights` must match that architecture. The default expects:

```text
weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The current adapter supports DINOv3 ViT backbones with patch tokens.
DINOv3 ConvNeXt backbones need a separate feature-map adapter and are not
included in this 2D baseline.

No pretrained weights are stored in this repository.

## Data

Training uses one 2D image and one integer instance-mask file per stem:

```text
data/train/images/sample_001.tif
data/train/masks/sample_001.tif
```

Supported inputs are `.npy`, `.png`, `.jpg`, `.tif`, and `.tiff`. Masks must
be 2D, use `0` for background, and positive integers for instances. Images may
be grayscale or contain one to three channels. Grayscale is mapped to
`[gray, 0, 0]`; each nonconstant channel is normalized by its 1st and 99th
percentiles without clipping.

Set the data paths and training options in `config/train.yaml`, then run:

```powershell
python run_train.py --device cuda
```

Each run under `run/<YYYYMMDD-HHMMSS>/` stores the effective configuration in
`train.yaml`, plain model weights in `model.pt`, the inference configuration in
`model.yaml`, training state in `checkpoint.pt`, and per-step loss in
`metrics.jsonl`. These existing weight/checkpoint filenames and the `weights/`
input directory are retained for compatibility. Existing run directories are
never overwritten by the CLI.

Settings use explicit CLI values, then YAML values, then code defaults.
Relative data, backbone, and resume paths in YAML are resolved against the
project root, even when the YAML is inside a run directory. This preserves
previous project-root invocations; callers formerly relying on another working
directory should supply absolute paths. CLI input/output paths remain relative
to the working directory. `--device` only overrides YAML when supplied.

Set `train.resume` or pass `--resume` to continue in a new run directory:

```powershell
python run_train.py --config run/<run-id>/train.yaml `
  --resume run/<run-id>/checkpoint.pt --device cuda
```

Increase `train.total_steps` in the selected configuration to train further.
Version 2 checkpoints restore model, optimizer, AMP scaler, completed step,
and Python/NumPy/PyTorch CPU/CUDA random states, and include the effective
configuration. Optimizer hyperparameters come from the checkpoint on resume
and are reflected in the saved configuration. The loader starts a fresh
shuffle: sampler position and worker/prefetch states are not restored, so
resume does **not** promise the same next examples or bitwise identical runs.
Legacy `{model, optimizer, step}` checkpoints are still accepted; their missing
scaler and random states start fresh.

## Inference

```powershell
python run_predict.py image.tif `
  --weights run/<run-id>/model.pt `
  --output masks.tif `
  --device cuda
```

Without `--config`, inference reads the adjacent `model.yaml`, then an adjacent
legacy `train.yaml`. Copy `model.pt` and `model.yaml` together when exporting.
Use `--config` explicitly for older standalone weight files; inference no longer
silently uses the current training YAML to interpret an old model. Configuration
format version 1 fixes the normalization and output contracts described here;
`predict` settings control tiling and mask reconstruction.

Inference normalizes the image, averages overlapping 384-pixel tiles, and runs
the Cellpose dynamics. The segmentation-only network has three output channels:

```text
0: Y flow, trained as 5 × unit flow
1: X flow, trained as 5 × unit flow
2: foreground logit
```

With `model.classes` enabled, separate class logits follow these three channels.
Inference additionally writes `<output-stem>.labels.json` with one class and
score per reconstructed mask. Classification weights use a version 2
`model.yaml` that preserves the ordered class names.

The default mask reconstruction uses a logit threshold of `0`, 200 flow steps,
a flow-consistency threshold of `0.4`, and a minimum mask size of 15 pixels.
PNG output supports labels through 65535; larger instance IDs require TIFF or
NumPy output. Tile accumulation uses float32 for float16/bfloat16 predictions
to avoid overflow, while the differentiable `run_tiled` function retains
input and parameter gradients.

## Source layout

```text
src/
  config.py              # settings, validation, project-relative paths
  build.py               # model, loader, optimizer, trainer assembly
  checkpoint.py          # plain inference state_dict IO
  model/                 # DINOv3 encoder and dense Cellpose head
  data/                  # file IO, pairing, dataset, random augmentation
  prepare/               # shared normalization, padding, mask-to-flow math
  train/                 # trainer, loss, resumable checkpoint state
  predict/               # inference, tile blending, flow-to-mask dynamics
```

Training and prediction share `prepare` calculations; training does not import
prediction code. Internal imports have moved: `src.dataset` to
`src.data.dataset`/`src.data.image`, `src.transforms` to `src.prepare.convert`,
`src.inference` to `src.predict.inference`/`src.predict.tile`, `src.dynamics` to
`src.prepare.flow`/`src.predict.dynamics`, and `src.train.engine` to
`src.train.trainer`. No internal compatibility aliases are kept.
See [architecture and upstream differences](docs/architecture.md) for the
preserved contracts and intentional deviations.

## Tests

The unit tests use a small fake DINO encoder, so they do not download weights:

```powershell
python -m pytest
python -m ruff check src tests scripts run_train.py run_predict.py
python -m ruff format --check src tests scripts run_train.py run_predict.py
```

The tests include offline flow references generated by the pinned upstream
Cellpose implementation and a synthetic train/resume/predict integration test.
Actual DINOv3 checks remain opt-in because the weights are gated and larger ViT
models need substantially more memory. During the refactor, the local ViT-B
checkpoint was also checked on an RTX 2060: strict legacy state loading, exact
before/after network output equality on a 32×40 input, CUDA AMP forward/backward,
and tiled prediction on a 23×37 image. This is an execution/compatibility check,
not a measurement of segmentation accuracy or full-size training throughput.

## Sources

The dense head, loss, flow targets, and mask dynamics are minimal adaptations
of Cellpose at commit
[`a54cb48`](https://github.com/MouseLand/cellpose/tree/a54cb48849b7e225a81e8e43dcb042d42427f543)
under its BSD 3-Clause license. The DINOv3 API was checked against commit
[`6876159`](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751)
and remains subject to Meta's separate license. See `THIRD_PARTY_NOTICES.md`.
