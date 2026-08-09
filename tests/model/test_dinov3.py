import pytest

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
