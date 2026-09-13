import torch
import torch.nn.functional as F

from .. import MODEL_STRIDE


def pad_image(
    image: torch.Tensor,
    tile_size: int,
    stride: int = MODEL_STRIDE,
) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    padded_height = max(tile_size, round_up(height, stride))
    padded_width = max(tile_size, round_up(width, stride))
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
