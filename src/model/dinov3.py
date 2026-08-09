from importlib import import_module
from pathlib import Path

from torch import nn

DINO_VIT_BACKBONES = (
    "vits16",
    "vits16plus",
    "vitb16",
    "vitl16",
    "vith16plus",
    "vit7b16",
)


def build_dinov3(
    weights: str | Path | None = None,
    model_name: str = "vitb16",
) -> nn.Module:
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")
    if model_name not in DINO_VIT_BACKBONES:
        choices = ", ".join(DINO_VIT_BACKBONES)
        raise ValueError(f"model_name must be one of: {choices}.")

    try:
        backbones = import_module("dinov3.hub.backbones")
    except ImportError as exc:
        raise ImportError(
            "DINOv3 is required. Install requirements.txt after accepting "
            "the DINOv3 license."
        ) from exc

    builder_name = f"dinov3_{model_name}"
    try:
        builder = getattr(backbones, builder_name)
    except AttributeError as exc:
        raise ImportError(f"installed DINOv3 does not expose {builder_name}.") from exc

    if weights is None:
        return builder(pretrained=False)

    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 backbone weights do not exist: {path}")
    return builder(pretrained=True, weights=str(path))
