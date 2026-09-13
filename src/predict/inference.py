from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..prepare.convert import normalize_image
from .classify import classify_instances
from .dynamics import compute_masks
from .tile import run_tiled


@dataclass(frozen=True)
class Prediction:
    masks: np.ndarray
    flow_logits: np.ndarray
    cell_logits: np.ndarray
    instances: tuple[dict, ...] = ()


def predict(
    model: nn.Module,
    image: np.ndarray,
    device: str | torch.device,
    channel_axis: int | None = None,
    tile_size: int = 384,
    overlap: float = 0.1,
    niter: int = 200,
    cellprob_threshold: float = 0.0,
    flow_threshold: float | None = 0.4,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    classes: tuple[str, ...] = (),
) -> Prediction:
    classes = tuple(classes)
    if len(classes) != getattr(model, "num_classes", 0):
        raise ValueError("classes must match the model classification head.")
    device = torch.device(device)
    normalized = normalize_image(image, channel_axis=channel_axis)
    inputs = torch.from_numpy(normalized).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = run_tiled(
                model,
                inputs,
                tile_size=tile_size,
                overlap=overlap,
            )
            flow_logits = output[0, :2].to("cpu", torch.float32).numpy().copy()
            cell_logits = output[0, 2].to("cpu", torch.float32).numpy().copy()
            class_logits = (
                output[0, 3:].to("cpu", torch.float32).numpy().copy()
                if classes
                else None
            )
    finally:
        model.train(was_training)
    masks = compute_masks(
        flow_logits,
        cell_logits,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        max_size_fraction=max_size_fraction,
        device=device,
    )
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.shape != cell_logits.shape:
        raise ValueError("computed masks must match the input spatial shape.")
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError("computed masks must use an integer dtype.")
    return Prediction(
        masks=masks,
        flow_logits=flow_logits,
        cell_logits=cell_logits,
        instances=classify_instances(masks, class_logits, classes) if classes else (),
    )
