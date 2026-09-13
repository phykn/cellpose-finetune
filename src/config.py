import math
from copy import deepcopy
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"
DEFAULTS = {
    "model": {
        "backbone": "vitb16",
        "backbone_weights": None,
        "patch_stride": 8,
        "classes": [],
    },
    "data": {
        "channel_axis": None,
        "crop_size": 384,
        "augment": True,
        "batch_size": 1,
        "num_workers": 0,
        "label_dir": None,
    },
    "optim": {"learning_rate": 1e-5, "weight_decay": 0.1},
    "train": {
        "total_steps": 10000,
        "mixed_precision": True,
        "save_every_steps": 500,
        "resume": None,
        "seed": 0,
        "weights": None,
        "class_loss_weight": 1.0,
    },
    "predict": {
        "tile_size": 384,
        "overlap": 0.1,
        "niter": 200,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "min_size": 15,
        "max_size_fraction": 0.4,
    },
}


def load_config(
    path: str | Path,
    device: str | None = None,
    resume: str | Path | None = None,
    weights: str | Path | None = None,
) -> dict:
    cfg = load_yaml(path)
    if cfg.get("format_version", 1) not in (1, 2):
        raise ValueError("unsupported configuration format_version.")
    for group, defaults in DEFAULTS.items():
        values = cfg.get(group, {})
        if not isinstance(values, dict):
            raise TypeError(f"{group} must be a mapping.")
        cfg[group] = deepcopy(defaults) | values
    cfg["device"] = (
        device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resume is not None:
        cfg["train"]["resume"] = str(resume)
        cfg["train"]["weights"] = None
    if weights is not None:
        cfg["train"]["weights"] = str(weights)
        if resume is None:
            cfg["train"]["resume"] = None
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    for group, key in (
        ("model", "backbone_weights"),
        ("data", "image_dir"),
        ("data", "mask_dir"),
        ("data", "label_dir"),
        ("train", "resume"),
        ("train", "weights"),
    ):
        value = cfg[group].get(key)
        if value is not None:
            path_value = Path(value).expanduser()
            if not path_value.is_absolute():
                path_value = PROJECT_ROOT / path_value
            cfg[group][key] = str(path_value.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    classes = cfg["model"]["classes"]
    if not isinstance(classes, list) or any(
        not isinstance(name, str) or not name.strip() for name in classes
    ):
        raise ValueError("model.classes must be a list of non-empty class names.")
    if len(set(classes)) != len(classes):
        raise ValueError("model.classes must not contain duplicate names.")
    if cfg["train"]["resume"] is not None and cfg["train"]["weights"] is not None:
        raise ValueError("use either train.resume or train.weights, not both.")
    weight = cfg["train"]["class_loss_weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("train.class_loss_weight must be finite and positive.")
    for group, key, minimum in (
        ("data", "batch_size", 1),
        ("data", "num_workers", 0),
        ("data", "crop_size", 16),
        ("train", "total_steps", 1),
        ("train", "save_every_steps", 1),
        ("train", "seed", 0),
        ("model", "patch_stride", 2),
        ("predict", "tile_size", 16),
        ("predict", "niter", 0),
        ("predict", "min_size", -1),
    ):
        value = cfg[group][key]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{group}.{key} must be an integer >= {minimum}.")
    stride = cfg["model"]["patch_stride"]
    if stride > 16 or stride % 2:
        raise ValueError("model.patch_stride must be even and between 2 and 16.")
    for group, key in (("data", "crop_size"), ("predict", "tile_size")):
        if cfg[group][key] % math.lcm(8, stride):
            raise ValueError(f"{group}.{key} must be divisible by 8 and patch_stride.")
    for group, key in (("data", "augment"), ("train", "mixed_precision")):
        if not isinstance(cfg[group][key], bool):
            raise TypeError(f"{group}.{key} must be a boolean.")
    for key in ("learning_rate", "weight_decay"):
        value = cfg["optim"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"optim.{key} must be a number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"optim.{key} must be finite and non-negative.")
    if cfg["train"]["seed"] >= 2**32:
        raise ValueError("train.seed must be less than 2**32.")
    for key in ("overlap", "cellprob_threshold", "flow_threshold", "max_size_fraction"):
        value = cfg["predict"][key]
        if key == "flow_threshold" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"predict.{key} must be a finite number.")
    pred = cfg["predict"]
    if not 0 <= pred["overlap"] < 1:
        raise ValueError("predict.overlap must be in [0, 1).")
    if not 0 < pred["max_size_fraction"] <= 1:
        raise ValueError("predict.max_size_fraction must be in (0, 1].")
    if pred["flow_threshold"] is not None and pred["flow_threshold"] < 0:
        raise ValueError("predict.flow_threshold must be non-negative or null.")


def find_prediction_config(weights: str | Path) -> Path:
    weights = Path(weights)
    for path in (weights.with_suffix(".yaml"), weights.parent / "train.yaml"):
        if path.is_file():
            return path
    raise FileNotFoundError("weights have no saved configuration; pass --config.")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return cfg


def save_yaml(path: str | Path, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)
