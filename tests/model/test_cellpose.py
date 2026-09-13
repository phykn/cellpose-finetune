import pytest
import torch
from torch import nn

from src.model.cellpose import CellposeDINO


class FakeEncoder(nn.Module):
    def __init__(self, channels: int = 12) -> None:
        super().__init__()
        self.num_features = channels
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, channels, 16, stride=16)
        self.scale = nn.Parameter(torch.ones(channels))

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.patch_embed.proj(image)
        tokens = features.flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": tokens * self.scale}


def test_model_restores_input_resolution_and_backpropagates() -> None:
    model = CellposeDINO(FakeEncoder())
    image = torch.randn(2, 3, 32, 40)

    output = model(image)
    output.square().mean().backward()

    assert output.shape == (2, 3, 32, 40)
    assert torch.isfinite(output).all()
    assert model.output.weight.grad is not None
    assert model.encoder.scale.grad is not None
    assert model.encoder.patch_embed.proj.weight.grad is None


@pytest.mark.parametrize("shape", [(3, 32, 32), (1, 1, 32, 32), (1, 3, 31, 32)])
def test_model_rejects_invalid_input_shape(shape: tuple[int, ...]) -> None:
    model = CellposeDINO(FakeEncoder())

    with pytest.raises(ValueError, match="image"):
        model(torch.randn(shape))


def test_model_rejects_non_vit16_encoder() -> None:
    encoder = FakeEncoder()
    encoder.patch_embed.proj = nn.Conv2d(3, 12, 14, stride=14)

    with pytest.raises(ValueError, match="ViT/16"):
        CellposeDINO(encoder)


@pytest.mark.parametrize("stride", [1, 3, 7, 15])
def test_model_rejects_stride_that_cannot_restore_resolution(stride):
    with pytest.raises(ValueError, match="even"):
        CellposeDINO(FakeEncoder(), patch_stride=stride)
