# Instance class labels (provisional format)

This extension uses full fine-tuning, without LoRA. Each instance has exactly
one class. It shares the DINO encoder between the original segmentation head
and a separate dense classification head. Annotation parsing lives in
`src/data/label.py` so a later format change need not change the network.

## Training inputs

```text
data/train/images/sample_001.tif
data/train/masks/sample_001.tif
data/train/labels/sample_001.json
```

Image and integer instance mask have matching spatial dimensions. Mask value
0 means background; positive IDs identify objects within that image. IDs may
be nonconsecutive and restart in each image. Different objects can share the
same class. The JSON maps the original mask IDs to names:

```json
{"2": "type_A", "9": "type_B", "15": "type_A"}
```

Set `model.classes: [type_A, type_B]` and `data.label_dir` in the configuration.
Names are case-sensitive strings; their order defines zero-based class IDs.
Background is not a classification category. Every positive ID in the original
mask must appear exactly once; extra IDs, unknown names and missing files/IDs
raise errors. An empty mask uses `{}`. This initial format requires complete
labels on every training image when classification is enabled.

The dataset validates IDs before crop/augmentation, preserves them through
paired transforms, and then constructs the class target. Background and padding
use ignore value -1. Targets have four channels: foreground, unit Y flow,
unit X flow, class ID. Without classes the original three-channel targets remain.

## Model and loss

Output channels are `[Y flow, X flow, foreground logit, K class logits]`.
The first three channels and segmentation tensor names remain unchanged.
The added `class_output` head performs the same dense pixel restoration as the
segmentation head. The loss is the original Cellpose loss plus
`train.class_loss_weight * foreground_class_cross_entropy` (default weight 1).
Only foreground pixels with labels supervise classification; empty crops have
zero classification loss. This initial loss weights foreground pixels equally,
so larger objects contribute more class supervision than smaller objects.

## Commands

Edit the example names and paths in `config/classify.yaml`, then run:

```powershell
python run_train.py --config config/classify.yaml --device cuda
```

To initialize from this project's older segmentation-only `model.pt`:

```powershell
python run_train.py --config config/classify.yaml `
  --weights run/<segmentation-run>/model.pt --device cuda
```

This strictly loads encoder/segmentation weights and initializes only the new
classification head. It starts a new optimizer; it is not checkpoint resume.
To resume a classification run, keep its class names/order and increase
`train.total_steps` in the selected configuration:

```powershell
python run_train.py --config run/<classification-run>/train.yaml `
  --resume run/<classification-run>/checkpoint.pt --device cuda
```

Explicit `--resume` replaces a saved initialization path; explicit `--weights`
replaces a saved resume path. Supplying both CLI options is rejected.
Resuming with reordered/different classes is rejected, even if the head shape
would otherwise match. Changing the class vocabulary needs a fresh head/run.

## Prediction outputs

```powershell
python run_predict.py image.tif --weights run/<classification-run>/model.pt `
  --output run/prediction/masks.tif --device cuda
```

Inference requires only the image and saved model/configuration. Copy `model.pt`
and its `model.yaml` together. Classification exports use configuration format
version 2; segmentation-only exports remain version 1. The sidecar contains
the ordered class names and prediction settings. Model assembly requires the
saved class mapping when loading weights that contain a classifier.
Raw `load_weights` remains a low-level tensor-loading API.

This writes `masks.tif` and `masks.labels.json`:

```json
{
  "format_version": 1,
  "classes": ["type_A", "type_B"],
  "instances": [
    {"mask_id": 1, "class_id": 0, "class_name": "type_A", "score": 0.94},
    {"mask_id": 2, "class_id": 1, "class_name": "type_B", "score": 0.87}
  ]
}
```

Each positive ID in the output mask has exactly one entry; an empty result has
`instances: []`. Prediction IDs need not match training IDs. Tiled class logits
are blended and restored to the original image size, converted to per-pixel
softmax probabilities, then averaged inside each final reconstructed mask.
The largest average chooses the class and its `score`. Scores are not calibrated
accuracy estimates. There is currently no unknown/rejection class or threshold.
The Python `Prediction.instances` field contains these same records.

With `model.classes: []` and `data.label_dir: null`, the existing segmentation-only
dataset, model, loss and mask-only CLI output remain supported. No class JSON
is written in that mode. Use distinct output paths when comparing the two modes.

Synthetic tests establish data alignment, gradients, checkpoint behavior and
mask-to-class association. Real cell-type classification accuracy still requires
training and evaluation on independent labeled specimens.
