import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

import run_predict
import run_train
from src.config import load_yaml, save_yaml
from src.model.cellpose import CellposeDINO


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 16, stride=16)

    def forward_features(self, image):
        return {
            "x_norm_patchtokens": self.patch_embed.proj(image)
            .flatten(2)
            .transpose(1, 2)
        }


@pytest.mark.parametrize("classification", [False, True])
def test_train_resume_and_predict_from_saved_configuration(
    tmp_path, monkeypatch, classification
):
    monkeypatch.setattr(
        "src.build.build_dinov3", lambda *args, **kwargs: SmallEncoder()
    )
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image_path = images / "cell.npy"
    np.save(image_path, np.arange(24 * 32, dtype=np.float32).reshape(24, 32))
    mask = np.zeros((24, 32), dtype=np.uint16)
    mask[4:12, 8:16] = 1
    np.save(masks / "cell.npy", mask)
    config = tmp_path / "source.yaml"
    cfg = {
        "device": "cpu",
        "data": {"image_dir": str(images), "mask_dir": str(masks), "crop_size": 16},
        "train": {"total_steps": 2, "save_every_steps": 1},
        "predict": {"tile_size": 16, "niter": 2},
    }
    if classification:
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "cell.json").write_text('{"1":"type_A"}')
        cfg["data"]["label_dir"] = str(labels)
        cfg["model"] = {"classes": ["type_A", "type_B"]}
        initial = tmp_path / "segmentation.pt"
        torch.save(CellposeDINO(SmallEncoder()).state_dict(), initial)
        cfg["train"]["weights"] = str(initial)
    save_yaml(config, cfg)
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(
        sys, "argv", ["run_train.py", "--config", str(config), "--run-dir", str(first)]
    )
    run_train.main()
    effective = load_yaml(first / "train.yaml")
    assert effective["device"] == "cpu"
    assert effective["model"]["patch_stride"] == 8
    assert len((first / "metrics.jsonl").read_text().splitlines()) == 2

    cfg["train"]["total_steps"] = 3
    cfg["optim"] = {"learning_rate": 0.5}
    save_yaml(config, cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_train.py",
            "--config",
            str(config),
            "--run-dir",
            str(second),
            "--resume",
            str(first / "checkpoint.pt"),
        ],
    )
    run_train.main()
    checkpoint = torch.load(second / "checkpoint.pt", weights_only=True)
    assert checkpoint["step"] == 3
    assert checkpoint["format_version"] == 2
    assert load_yaml(second / "train.yaml")["optim"]["learning_rate"] == 1e-5
    metrics = json.loads((second / "metrics.jsonl").read_text())
    assert metrics["step"] == 3

    # A changed source YAML must not alter inference for already exported weights.
    save_yaml(config, {"model": {"patch_stride": 16}})
    output = tmp_path / "prediction.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_predict.py",
            str(image_path),
            "--weights",
            str(second / "model.pt"),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    run_predict.main()
    result = np.load(output)
    assert result.shape == mask.shape
    assert np.issubdtype(result.dtype, np.integer)
    if classification:
        saved = load_yaml(second / "model.yaml")
        assert saved["format_version"] == 2
        assert saved["model"]["classes"] == ["type_A", "type_B"]
        labels = json.loads(output.with_suffix(".labels.json").read_text())
        assert labels["classes"] == ["type_A", "type_B"]
        assert {entry["mask_id"] for entry in labels["instances"]} == set(
            np.unique(result)
        ) - {0}
