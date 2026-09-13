import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    scaler: torch.amp.GradScaler | None = None,
    cfg: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": cfg,
            "rng": capture_rng(),
        },
        temp,
    )
    os.replace(temp, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    scaler: torch.amp.GradScaler | None = None,
    cfg: dict | None = None,
) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    version = state.get("format_version", 1)
    fields = {"model", "optimizer", "step"}
    if version == 2:
        fields |= {"format_version", "scaler", "config", "rng"}
    elif version != 1:
        raise ValueError(f"unsupported checkpoint format version: {version}")
    if set(state) != fields:
        raise ValueError(f"checkpoint has unexpected fields: {path}")
    step = state["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"checkpoint step must be a non-negative integer: {path}")
    saved_cfg = state.get("config")
    if cfg is not None and saved_cfg is not None:
        if cfg["model"].get("classes", []) != saved_cfg["model"].get("classes", []):
            raise ValueError(
                "resume model.classes differs from saved class names/order."
            )
        for key in ("backbone", "patch_stride"):
            if cfg["model"].get(key) != saved_cfg["model"].get(key):
                raise ValueError(
                    f"resume model.{key} differs from saved configuration."
                )
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and scaler.is_enabled() and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if version == 2:
        restore_rng(state["rng"])
    return step


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state(
        (name, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached)
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        for idx, rng in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng.cpu(), device=idx)
