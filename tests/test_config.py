import pytest

from src.config import PROJECT_ROOT, find_prediction_config, load_config, save_yaml


def test_config_resolves_paths_and_preserves_explicit_precedence(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    save_yaml(path, {"device": "cpu", "data": {"image_dir": "data/images"}})
    monkeypatch.chdir(tmp_path)
    cfg = load_config(path)
    assert cfg["device"] == "cpu"
    assert cfg["data"]["image_dir"] == str(PROJECT_ROOT / "data" / "images")
    assert cfg["data"]["crop_size"] == 384
    assert cfg["model"]["patch_stride"] == 8
    save_yaml(path, {"device": "cuda", "train": {"resume": "old.pt"}})
    cfg = load_config(path, device="cpu", resume="new.pt")
    assert cfg["device"] == "cpu"
    assert cfg["train"]["resume"] == str(PROJECT_ROOT / "new.pt")


@pytest.mark.parametrize(
    "cfg",
    [
        {"model": {"patch_stride": 3}},
        {"model": {"patch_stride": 16}, "data": {"crop_size": 24}},
        {"train": {"mixed_precision": "false"}},
        {"data": {"batch_size": 0}},
    ],
)
def test_invalid_configuration_fails_before_building(cfg, tmp_path):
    path = tmp_path / "train.yaml"
    save_yaml(path, cfg)
    with pytest.raises((TypeError, ValueError)):
        load_config(path, device="cpu")


def test_prediction_uses_weights_sidecar_or_legacy_run_config(tmp_path):
    weights = tmp_path / "model.pt"
    with pytest.raises(FileNotFoundError, match="--config"):
        find_prediction_config(weights)
    legacy = tmp_path / "train.yaml"
    save_yaml(legacy, {})
    assert find_prediction_config(weights) == legacy
    sidecar = tmp_path / "model.yaml"
    save_yaml(sidecar, {})
    assert find_prediction_config(weights) == sidecar
