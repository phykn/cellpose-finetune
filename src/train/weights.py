import os
from pathlib import Path

import torch
from torch import nn


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
) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights do not exist: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"model weights must contain a state dict: {path}")
    model.load_state_dict(state, strict=True)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        temp,
    )
    os.replace(temp, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    if set(state) != {"model", "optimizer", "step"}:
        raise ValueError(f"checkpoint has unexpected fields: {path}")
    step = state["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"checkpoint step must be a non-negative integer: {path}")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    return step
