# DINOv3 Cellpose

Cellpose turns a cell image into two inward flow fields and one foreground
logit, then follows those flows to recover individual masks. This repository
keeps only that 2D path and uses a DINOv3 ViT encoder, with ViT-B/16 as the
default.

This first milestone intentionally contains no LoRA code. It establishes the
plain DINOv3 Cellpose baseline that LoRA can be compared against later. SAM,
the legacy U-Net, 3D inference, the GUI, restoration, model-zoo downloads, and
distributed training are not included.

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

Each run stores its config, the latest plain model state in `model.pt`, and a
resumable optimizer state in `checkpoint.pt`. Set `train.resume` to a
`checkpoint.pt` path to continue a run in a new output directory.

## Inference

```powershell
python run_predict.py image.tif `
  --weights run/<run-id>/model.pt `
  --config run/<run-id>/train.yaml `
  --output masks.tif `
  --device cuda
```

Inference normalizes the image, averages overlapping 384-pixel tiles, and runs
the Cellpose dynamics. The network contract is exactly three output channels:

```text
0: Y flow, trained as 5 × unit flow
1: X flow, trained as 5 × unit flow
2: foreground logit
```

The default mask reconstruction uses a logit threshold of `0`, 200 flow steps,
a flow-consistency threshold of `0.4`, and a minimum mask size of 15 pixels.

## Tests

The unit tests use a small fake DINO encoder, so they do not download weights:

```powershell
python -m pytest
python -m ruff check src tests run_train.py run_predict.py
```

An actual DINOv3 checkpoint smoke test is deliberately opt-in because the
weights are gated and larger ViT models need substantially more memory.

## Sources

The dense head, loss, flow targets, and mask dynamics are minimal adaptations
of Cellpose at commit
[`a54cb48`](https://github.com/MouseLand/cellpose/tree/a54cb48849b7e225a81e8e43dcb042d42427f543)
under its BSD 3-Clause license. The DINOv3 API was checked against commit
[`6876159`](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751)
and remains subject to Meta's separate license. See `THIRD_PARTY_NOTICES.md`.
