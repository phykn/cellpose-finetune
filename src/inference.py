import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import MODEL_STRIDE
from .dynamics import compute_masks
from .transforms import normalize_image


@dataclass(frozen=True)
class Prediction:
    masks: np.ndarray
    flow_logits: np.ndarray
    cell_logits: np.ndarray


def run_tiled(
    model: nn.Module,
    image: torch.Tensor,
    tile_size: int = 384,
    overlap: float = 0.1,
) -> torch.Tensor:
    check_tiled_input(image, tile_size, overlap)
    height, width = image.shape[-2:]
    padded, top, left = pad_image(image, tile_size)
    padded_height, padded_width = padded.shape[-2:]
    rows = get_tile_starts(padded_height, tile_size, overlap)
    cols = get_tile_starts(padded_width, tile_size, overlap)

    output_sum = None
    weight_sum = None
    for row in rows:
        for col in cols:
            tile = padded[..., row : row + tile_size, col : col + tile_size]
            output = model(tile)
            check_tiled_output(output, tile_size)
            weight = make_taper(tile_size, output)
            padding = (
                col,
                padded_width - col - tile_size,
                row,
                padded_height - row - tile_size,
            )
            weighted = F.pad(output * weight, padding)
            placed_weight = F.pad(weight, padding)
            output_sum = weighted if output_sum is None else output_sum + weighted
            weight_sum = (
                placed_weight if weight_sum is None else weight_sum + placed_weight
            )

    if output_sum is None or weight_sum is None:
        raise RuntimeError("tiled inference produced no output.")
    output = output_sum / weight_sum
    return output[..., top : top + height, left : left + width]


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
) -> Prediction:
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
    )


def check_tiled_input(
    image: torch.Tensor,
    tile_size: int,
    overlap: float,
) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have shape [1, 3, H, W].")
    if image.shape[-2] < 1 or image.shape[-1] < 1:
        raise ValueError("image spatial dimensions must be positive.")
    if not image.is_floating_point():
        raise TypeError("image must use a floating-point dtype.")
    if (
        not isinstance(tile_size, int)
        or isinstance(tile_size, bool)
        or tile_size < 2 * MODEL_STRIDE
        or tile_size % MODEL_STRIDE
    ):
        raise ValueError(
            f"tile_size must be divisible by {MODEL_STRIDE} and at least "
            f"{2 * MODEL_STRIDE}."
        )
    if (
        not isinstance(overlap, (int, float))
        or isinstance(overlap, bool)
        or not math.isfinite(overlap)
        or not 0.0 <= overlap < 1.0
    ):
        raise ValueError("overlap must be between zero inclusive and one exclusive.")


def check_tiled_output(output: object, tile_size: int) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError("model must return a torch.Tensor.")
    expected = (1, 3, tile_size, tile_size)
    if output.shape != expected:
        raise ValueError(f"model output must have shape {expected}.")
    if not output.is_floating_point():
        raise TypeError("model output must use a floating-point dtype.")


def pad_image(
    image: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    padded_height = max(tile_size, round_up(height, MODEL_STRIDE))
    padded_width = max(tile_size, round_up(width, MODEL_STRIDE))
    pad_height = padded_height - height
    pad_width = padded_width - width
    top = pad_height // 2
    left = pad_width // 2
    padded = F.pad(
        image,
        (
            left,
            pad_width - left,
            top,
            pad_height - top,
        ),
    )
    return padded, top, left


def round_up(value: int, stride: int) -> int:
    return ((value + stride - 1) // stride) * stride


def get_tile_starts(
    size: int,
    tile_size: int,
    overlap: float,
) -> tuple[int, ...]:
    if size <= tile_size:
        return (0,)
    step = round(tile_size * (1.0 - overlap) / MODEL_STRIDE) * MODEL_STRIDE
    step = min(tile_size, max(MODEL_STRIDE, step))
    last = size - tile_size
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def make_taper(tile_size: int, reference: torch.Tensor) -> torch.Tensor:
    position = torch.linspace(
        -math.pi / 2.0,
        math.pi / 2.0,
        tile_size,
        device=reference.device,
        dtype=torch.float32,
    )
    edge = position.cos().square()
    weight = (edge[:, None] * edge[None, :]).clamp_min(1.0e-3)
    return weight.to(dtype=reference.dtype)[None, None]
