import random

import numpy as np
import pytest
import torch
from torch import nn

from src.train.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_restores_rng_scaler_and_next_optimizer_update(tmp_path):
    torch.manual_seed(19)
    model = nn.Sequential(nn.Dropout(0.5), nn.Linear(3, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    image = torch.ones(2, 3)

    def step():
        optimizer.zero_grad()
        model(image).square().mean().backward()
        optimizer.step()

    step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 1, scaler=scaler)
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}

    restored_scaler = torch.amp.GradScaler("cpu", init_scale=32)
    assert load_checkpoint(path, model, optimizer, scaler=restored_scaler) == 1
    assert restored_scaler.state_dict() == scaler.state_dict()
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_legacy_checkpoint_is_still_readable(tmp_path):
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "legacy.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 7},
        path,
    )
    assert load_checkpoint(path, model, optimizer) == 7


def test_resume_rejects_same_shape_different_stride_before_loading(tmp_path):
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        0,
        cfg={"model": {"backbone": "vitb16", "patch_stride": 8}},
    )
    with pytest.raises(ValueError, match="patch_stride"):
        load_checkpoint(
            path,
            model,
            optimizer,
            cfg={"model": {"backbone": "vitb16", "patch_stride": 16}},
        )
