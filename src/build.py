from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import CellposeDataset
from .model.cellpose import CellposeDINO
from .model.dinov3 import build_dinov3
from .train.engine import BatchStream, Trainer
from .train.loss import CellposeLoss
from .train.weights import load_checkpoint, load_weights


def build_model(
    cfg: dict,
    weights: str | Path | None = None,
    load_backbone: bool = True,
) -> CellposeDINO:
    model_cfg = cfg["model"]
    backbone_weights = model_cfg.get("backbone_weights") if load_backbone else None
    encoder = build_dinov3(
        backbone_weights,
        model_name=model_cfg.get("backbone", "vitb16"),
    )
    model = CellposeDINO(
        encoder,
        patch_stride=model_cfg.get("patch_stride", 8),
    )
    if weights is not None:
        load_weights(weights, model)
    return model


def build_dataset(cfg: dict) -> CellposeDataset:
    data = cfg["data"]
    return CellposeDataset(
        data["image_dir"],
        data["mask_dir"],
        crop_size=data.get("crop_size", 384),
        channel_axis=data.get("channel_axis"),
        augment=data.get("augment", True),
    )


def build_loader(cfg: dict, device: torch.device) -> DataLoader:
    data = cfg["data"]
    num_workers = data.get("num_workers", 0)
    if not isinstance(num_workers, int) or isinstance(num_workers, bool):
        raise TypeError("data.num_workers must be an integer.")
    if num_workers < 0:
        raise ValueError("data.num_workers must be non-negative.")
    batch_size = data.get("batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("data.batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    return DataLoader(
        build_dataset(cfg),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def build_trainer(cfg: dict, device: torch.device) -> Trainer:
    train = cfg["train"]
    resume = train.get("resume")
    model = build_model(cfg, load_backbone=resume is None).to(device)
    optim = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim.get("learning_rate", 1e-5),
        weight_decay=optim.get("weight_decay", 0.1),
    )
    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(resume, model, optimizer, device=device)
    return Trainer(
        model,
        BatchStream(build_loader(cfg, device)),
        optimizer,
        CellposeLoss(),
        device,
        mixed_precision=train.get("mixed_precision", True),
        start_step=start_step,
    )
