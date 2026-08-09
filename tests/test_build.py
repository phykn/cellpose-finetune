import torch
from torch import nn

from src import build


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 4
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 16, stride=16)

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.patch_embed.proj(image)
        return {"x_norm_patchtokens": features.flatten(2).transpose(1, 2)}


def test_build_model_passes_backbone_options(monkeypatch, tmp_path) -> None:
    requested = []
    monkeypatch.setattr(
        build,
        "build_dinov3",
        lambda weights, model_name: (
            requested.append((weights, model_name)) or FakeEncoder()
        ),
    )
    cfg = {
        "model": {
            "backbone": "vitl16",
            "backbone_weights": str(tmp_path / "backbone.pt"),
            "patch_stride": 8,
        }
    }

    model = build.build_model(cfg)

    assert requested == [(str(tmp_path / "backbone.pt"), "vitl16")]
    assert model.patch_stride == 8
