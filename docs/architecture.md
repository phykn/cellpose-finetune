# Cellpose DINO architecture and refactor contract

The layout follows the responsibility boundaries in `D:/code/guide.md` and the
user wiki's code/refactor guidance. Existing CLI options, plain model tensor
names, configuration group names, and run weight filenames are retained.
The source wiki is not modified by this refactor.

## Ownership

`run_train.py` resolves configuration and seeds, assembles objects through
`src.build`, allocates a run directory, and invokes `Trainer.fit`.
`run_predict.py` selects the saved inference configuration, assembles the model,
decodes the input, invokes prediction, and saves the integer mask.

`data.image` owns file decoding, encoding, and stem pairing. `data.dataset`
reads each pair and invokes common normalization, paired crop/augmentation,
and flow-target generation in that order. `prepare.flow` owns the shared
mask-to-flow conversion used both to create training targets and to check
predicted masks. `predict.dynamics` owns endpoint integration, instance
construction, flow consistency filtering, hole filling, and size filtering.
Putting all flow math in `predict` would make the training dataset depend on
the inference layer; putting it all in `data` would couple inference to loading.

`prepare.resize` returns padded tensors and top/left offsets for restoration.
`predict.tile` selects stride-aligned tile locations, blends predictions, and
crops back to the original input extent. Accumulators cover the padded image
once; each tile updates a slice rather than allocating another full-image
tensor. Overlap sums use at least float32. `predict.inference` owns eval and
inference mode and restores the previous model mode even if forward fails.

`checkpoint.py` handles plain state dictionaries shared by assembly and training.
`train.checkpoint` handles optimizer, scaler, step, configuration and RNG state.
Segmentation-only tensor names and shapes are unchanged. The checkpoint reader accepts
both the original three-field training checkpoint and version 2. Older code
cannot read version 2 training state, but can still read the plain `model.pt`.

`model.yaml` configuration format version 1 records backbone, stride, channel
axis, and prediction parameters alongside plain weights. Version 1 implies the
fixed preprocessing/output rules below. A future change to those rules requires
a version change and migration, rather than reinterpreting existing metadata.
The sidecar omits the original backbone file path because `model.pt` contains
the complete trained encoder. Moving weights requires moving their sidecar.

Optional instance classification adds `class_output` parameters and K output
channels after the original three segmentation channels. Its version 2
configuration also records the ordered class vocabulary. Classification loss
is calculated on labeled foreground pixels; prediction aggregates class
probabilities inside final masks. See [the provisional label contract](labels.md)
for annotation, initialization/resume and JSON output details. Ordinary
fine-tuning is the supported training approach; no LoRA is planned here.

## Upstream and preserved improvements

The reference is [Cellpose a54cb488](https://github.com/MouseLand/cellpose/tree/a54cb48849b7e225a81e8e43dcb042d42427f543),
also upstream HEAD when checked on 2026-09-13.

| Contract | Reference / local behavior |
|---|---|
| Network | `cellpose/vit.py:CPDINO`: ViT/16 projection with stride 8 and padding 4 by default; projection parameters frozen; three dense channels |
| Dense restoration | Local pixel shuffle equals upstream fixed identity transposed convolution in both values and gradients |
| Channels | Prediction `[Y flow, X flow, foreground logit]`; target `[foreground, unit Y flow, unit X flow]` |
| Loss | `cellpose/train.py:_loss_fn_seg`: mean flow MSE against 5×target, divided by 2, plus mean BCE with logits |
| Target flows | `cellpose/dynamics.py:masks_to_flows_gpu`: nine-neighbor heat diffusion and normalized Y/X differences; centers rounded in each instance bounding box |
| Input | One to three image channels, percentile 1/99 normalization without clipping; grayscale becomes `[gray, 0, 0]`; constant channels become zero |
| Reconstruction | Foreground logit threshold, flow divided by 5, bilinear Euler integration, endpoint clustering, flow QC and size filtering |

The repository remains a 2D baseline without LoRA. It keeps the official
DINOv3 `forward_features` interface, supports more ViT sizes than upstream's
B/L selection, and obtains normalized patch tokens without hardcoding the
number of storage tokens. It keeps pixel shuffle instead of a saved fixed
identity kernel. The default stride 8 model has unchanged tensor names and
numerical network output. Other even strides through 16 are supported; crop
and tile sizes must be divisible by both 8 and the selected stride. Odd strides
are rejected at construction because their padding cannot restore input size.

This baseline intentionally retains its paired 90-degree rotations/flips,
stride-aligned cosine tile taper, and NumPy/SciPy 2D flow implementation.
It does not add upstream stochastic block dropping, full augmentation policies,
model-zoo state conversion, or segmentation performance claims. Upstream CPDINO
checkpoints use different head keys and are not interchangeable with this
repository's complete model state dictionaries.

## Verified corrections

- Instance-center rounding now uses local bounding-box coordinates. Previously,
  a half-integer center in a box beginning on an odd coordinate could choose a
  different pixel than upstream. This intentionally changes those flow targets
  and may affect reconstructed masks; the network and loss contract is unchanged.
- PNG export rejects instance IDs above 65535 rather than wrapping to background
  or a different instance. TIFF and NumPy retain larger labels.
- Local DINO weights are read directly with strict loading rather than through
  the hub's filename cache, which can confuse different files sharing a basename.
- Tile accumulation avoids half-precision overlap overflow and repeated
  full-image allocations, and respects the actual model stride.
- Resumed training restores scaler and process RNG state; a run already at the
  requested final step still exports weights. Data order/prefetch resume is not
  implemented. Saved settings record restored optimizer hyperparameters.
- Explicit device overrides take precedence over saved settings, and inference
  defaults to the weight's saved configuration rather than today's training YAML.

## Evidence and limits

Offline fixtures are generated from the pinned upstream flow functions on
synthetic rectangles, concave masks, borders and empty images. See
`tests/fixtures/README.md` for regeneration. Tests also cover dense-head values
and gradients, flow/mask round trips, tile blending and gradients, plain and
legacy training weights, scaler/RNG restoration with an identical subsequent
optimizer update, and the CLI train/resume/predict path.

The CPU NumPy implementation is not claimed to be bitwise identical to upstream
PyTorch on all masks/devices: summation order can change the direction of
near-zero gradients at symmetric centers. Fixture comparisons use a 1e-6
absolute tolerance; mask round trips are checked separately. Real-data quality,
384-pixel training memory/throughput, and multiworker replay are not established
by small synthetic tests or the recorded ViT-B CUDA smoke check.
