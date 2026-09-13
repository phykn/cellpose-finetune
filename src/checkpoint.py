import os
from pathlib import Path

import torch
from torch import nn

from .config import find_prediction_config, load_yaml


def save_weights(path: str | Path, model: nn.Module) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temp)
    os.replace(temp, path)


def load_weights(
    path: str | Path,
    model: nn.Module,
    device: torch.device | str = "cpu",
    initialize_classifier: bool = False,
    classes: list[str] | None = None,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights do not exist: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"model weights must contain a state dict: {path}")
    if classes is not None and any(key.startswith("class_output.") for key in state):
        saved = load_yaml(find_prediction_config(path))
        if saved["model"].get("classes", []) != classes:
            raise ValueError("model.classes differs from the saved class names/order.")
    if (
        initialize_classifier
        and getattr(model, "num_classes", 0)
        and not any(key.startswith("class_output.") for key in state)
    ):
        # Only the newly added classifier may be absent. Still load everything strictly.
        state = dict(state)
        state.update(
            {
                key: value
                for key, value in model.state_dict().items()
                if key.startswith("class_output.")
            }
        )
    model.load_state_dict(state, strict=True)
