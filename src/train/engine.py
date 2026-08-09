from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .weights import save_checkpoint, save_weights


class BatchStream:
    def __init__(self, loader: DataLoader) -> None:
        if len(loader) == 0:
            raise ValueError("training loader must contain at least one batch.")
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        stream: BatchStream,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        mixed_precision: bool = True,
        start_step: int = 0,
    ) -> None:
        if not isinstance(mixed_precision, bool):
            raise TypeError("mixed_precision must be a boolean.")
        if not isinstance(start_step, int) or isinstance(start_step, bool):
            raise TypeError("start_step must be an integer.")
        if start_step < 0:
            raise ValueError("start_step must be non-negative.")
        self.model = model
        self.stream = stream
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.amp_enabled = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.step_idx = start_step

    def fit(self, steps: int, save_every: int, run_dir: str | Path) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(save_every, int) or isinstance(save_every, bool):
            raise TypeError("save_every must be an integer.")
        if save_every <= 0:
            raise ValueError("save_every must be positive.")
        if self.step_idx > steps:
            raise ValueError("checkpoint step exceeds requested total steps.")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        while self.step_idx < steps:
            loss = self.step()
            self.step_idx += 1
            if self.step_idx % save_every == 0 or self.step_idx == steps:
                self.save(run_dir)
                print(f"step={self.step_idx} loss={loss:.6f}")

    def step(self) -> float:
        self.model.train()
        image, target = self.stream.next()
        image = image.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            prediction = self.model(image)
            loss = self.criterion(prediction, target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach())

    def save(self, run_dir: Path) -> None:
        save_weights(run_dir / "model.pt", self.model)
        save_checkpoint(
            run_dir / "checkpoint.pt",
            self.model,
            self.optimizer,
            self.step_idx,
        )
