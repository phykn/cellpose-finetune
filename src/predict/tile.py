import math

import torch
from torch import nn

from .. import MODEL_STRIDE
from ..prepare.resize import pad_image


def run_tiled(
    model: nn.Module,
    image: torch.Tensor,
    tile_size: int = 384,
    overlap: float = 0.1,
) -> torch.Tensor:
    check_tiled_input(image, tile_size, overlap)
    output_channels = 3 + getattr(model, "num_classes", 0)
    stride = math.lcm(MODEL_STRIDE, getattr(model, "patch_stride", MODEL_STRIDE))
    if tile_size % stride:
        raise ValueError("tile_size must be divisible by the model patch_stride.")
    height, width = image.shape[-2:]
    padded, top, left = pad_image(image, tile_size, stride=stride)
    padded_height, padded_width = padded.shape[-2:]
    rows = get_tile_starts(padded_height, tile_size, overlap, stride=stride)
    cols = get_tile_starts(padded_width, tile_size, overlap, stride=stride)

    output_sum = None
    weight_sum = None
    weight = None
    for row in rows:
        for col in cols:
            tile = padded[..., row : row + tile_size, col : col + tile_size]
            output = model(tile)
            check_tiled_output(output, tile_size, output_channels)
            if output_sum is None:
                # Accumulate half-precision predictions in float32 across overlaps.
                dtype = (
                    torch.float64 if output.dtype == torch.float64 else torch.float32
                )
                output_sum = torch.zeros(
                    (1, output_channels, padded_height, padded_width),
                    device=output.device,
                    dtype=dtype,
                )
                weight_sum = torch.zeros_like(output_sum[:, :1])
                weight = make_taper(tile_size, output_sum)
            region = (..., slice(row, row + tile_size), slice(col, col + tile_size))
            output_sum[region] += output * weight
            weight_sum[region] += weight

    if output_sum is None or weight_sum is None:
        raise RuntimeError("tiled inference produced no output.")
    output = output_sum / weight_sum
    return output[..., top : top + height, left : left + width]


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


def check_tiled_output(output: object, tile_size: int, channels: int = 3) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError("model must return a torch.Tensor.")
    expected = (1, channels, tile_size, tile_size)
    if output.shape != expected:
        raise ValueError(f"model output must have shape {expected}.")
    if not output.is_floating_point():
        raise TypeError("model output must use a floating-point dtype.")


def get_tile_starts(
    size: int,
    tile_size: int,
    overlap: float,
    stride: int = MODEL_STRIDE,
) -> tuple[int, ...]:
    if size <= tile_size:
        return (0,)
    step = round(tile_size * (1.0 - overlap) / stride) * stride
    step = min(tile_size, max(stride, step))
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
