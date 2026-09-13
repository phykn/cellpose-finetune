from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model.dinov3 import DINO_VIT_BACKBONES, build_dinov3


def test_supported_dino_vit_backbones_are_explicit() -> None:
    assert DINO_VIT_BACKBONES == (
        "vits16",
        "vits16plus",
        "vitb16",
        "vitl16",
        "vith16plus",
        "vit7b16",
    )


def test_unknown_dino_backbone_fails_before_import() -> None:
    with pytest.raises(ValueError, match="model_name must be one of"):
        build_dinov3(model_name="convnext_tiny")


def test_local_backbone_weights_bypass_hub_filename_cache(monkeypatch, tmp_path):
    calls = []

    def builder(pretrained):
        calls.append(pretrained)
        return nn.Linear(2, 1)

    monkeypatch.setattr(
        "src.model.dinov3.import_module",
        lambda _: SimpleNamespace(dinov3_vitb16=builder),
    )
    for value, folder in enumerate(("first", "second"), start=1):
        path = tmp_path / folder / "model.pth"
        path.parent.mkdir()
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(value)
        torch.save(model.state_dict(), path)
        restored = build_dinov3(path)
        torch.testing.assert_close(restored.weight, model.weight)
    assert calls == [False, False]
