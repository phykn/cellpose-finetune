import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.checkpoint import load_weights
from src.train.checkpoint import load_checkpoint
from src.train.trainer import BatchStream, Trainer


def make_trainer() -> tuple[Trainer, nn.Module, torch.optim.Optimizer]:
    model = nn.Conv2d(3, 3, 1)
    images = torch.randn(2, 3, 8, 8)
    targets = torch.zeros_like(images)
    loader = DataLoader(TensorDataset(images, targets), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer = Trainer(
        model,
        BatchStream(loader),
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
        mixed_precision=False,
    )
    return trainer, model, optimizer


def test_trainer_runs_and_round_trips_checkpoint(tmp_path) -> None:
    trainer, model, _ = make_trainer()

    trainer.fit(steps=2, save_every=1, run_dir=tmp_path)

    assert trainer.step_idx == 2
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "checkpoint.pt").is_file()

    restored_trainer, restored_model, restored_optimizer = make_trainer()
    load_weights(tmp_path / "model.pt", restored_model)
    for actual, expected in zip(
        restored_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        assert torch.equal(actual, expected)

    step = load_checkpoint(
        tmp_path / "checkpoint.pt",
        restored_model,
        restored_optimizer,
    )
    assert step == 2
    assert restored_trainer.step_idx == 0


def test_completed_resume_still_exports_weights(tmp_path):
    trainer, _, _ = make_trainer()
    trainer.step_idx = 2
    trainer.fit(steps=2, save_every=1, run_dir=tmp_path)
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "checkpoint.pt").is_file()
