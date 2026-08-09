from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .. import MODEL_STRIDE


class CellposeDINO(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        patch_stride: int = MODEL_STRIDE,
    ) -> None:
        super().__init__()
        if not isinstance(patch_stride, int) or isinstance(patch_stride, bool):
            raise TypeError("patch_stride must be an integer.")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive.")

        try:
            projection = encoder.patch_embed.proj
        except AttributeError as exc:
            raise TypeError("encoder must expose DINOv3 patch_embed.proj.") from exc
        if not isinstance(projection, nn.Conv2d):
            raise TypeError("encoder.patch_embed.proj must be Conv2d.")
        if projection.kernel_size != (16, 16):
            raise ValueError("only DINOv3 ViT/16 encoders are supported.")
        if patch_stride > projection.kernel_size[0]:
            raise ValueError("patch_stride must not exceed the patch size.")

        projection.stride = (patch_stride, patch_stride)
        padding = (projection.kernel_size[0] - patch_stride) // 2
        projection.padding = (padding, padding)
        projection.requires_grad_(False)

        channels = getattr(encoder, "num_features", None)
        if not isinstance(channels, int) or channels <= 0:
            channels = getattr(encoder, "embed_dim", None)
        if not isinstance(channels, int) or channels <= 0:
            raise TypeError("encoder must expose a positive feature dimension.")

        self.encoder = encoder
        self.output = nn.Linear(channels, 3 * patch_stride**2)
        self.patch_stride = patch_stride

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width].")
        height, width = image.shape[-2:]
        if height < 16 or width < 16:
            raise ValueError("image height and width must be at least 16.")
        if height % self.patch_stride or width % self.patch_stride:
            raise ValueError(
                f"image height and width must be divisible by {self.patch_stride}."
            )

        features = self.encoder.forward_features(image)
        tokens = self.get_patch_tokens(features)
        grid_height, grid_width = self.get_grid_shape(height, width)
        if tokens.shape[1] != grid_height * grid_width:
            raise RuntimeError(
                "DINOv3 patch token count does not match the patch grid."
            )

        output = self.output(tokens)
        output = output.reshape(
            image.shape[0],
            grid_height,
            grid_width,
            3 * self.patch_stride**2,
        )
        output = output.permute(0, 3, 1, 2).contiguous()
        output = F.pixel_shuffle(output, self.patch_stride)
        if output.shape[-2:] != (height, width):
            raise RuntimeError("dense head did not restore the input resolution.")
        return output

    @staticmethod
    def get_patch_tokens(features: object) -> torch.Tensor:
        if not isinstance(features, Mapping):
            raise TypeError("DINOv3 forward_features must return a mapping.")
        tokens = features.get("x_norm_patchtokens")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise TypeError(
                "DINOv3 features must contain [batch, patches, channels] "
                "x_norm_patchtokens."
            )
        return tokens

    def get_grid_shape(self, height: int, width: int) -> tuple[int, int]:
        projection = self.encoder.patch_embed.proj
        kernel_height, kernel_width = projection.kernel_size
        stride_height, stride_width = projection.stride
        pad_height, pad_width = projection.padding
        grid_height = (height + 2 * pad_height - kernel_height) // stride_height + 1
        grid_width = (width + 2 * pad_width - kernel_width) // stride_width + 1
        return grid_height, grid_width
