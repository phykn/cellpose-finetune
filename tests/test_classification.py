import numpy as np
import pytest
import torch

from src import FLOW_SCALE, build
from src.checkpoint import load_weights, save_weights
from src.config import save_yaml
from src.data.dataset import CellposeDataset
from src.data.label import read_labels
from src.model.cellpose import CellposeDINO
from src.predict.classify import classify_instances
from src.predict.inference import predict
from src.predict.tile import run_tiled
from src.prepare.flow import flow_targets
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.loss import CellposeLoss
from tests.model.test_cellpose import FakeEncoder


def test_class_targets_follow_original_ids_through_random_augmentation(tmp_path):
    for folder in ("images", "masks", "labels"):
        (tmp_path / folder).mkdir()
    mask = np.zeros((24, 32), dtype=np.uint16)
    mask[2:12, 3:13] = 2
    mask[12:22, 17:27] = 9
    np.save(tmp_path / "images" / "cell.npy", mask.astype(np.float32))
    np.save(tmp_path / "masks" / "cell.npy", mask)
    (tmp_path / "labels" / "cell.json").write_text('{"2":"A", "9":"B"}')
    dataset = CellposeDataset(
        tmp_path / "images",
        tmp_path / "masks",
        crop_size=16,
        label_dir=tmp_path / "labels",
        classes=("A", "B"),
    )
    torch.manual_seed(17)
    for _ in range(8):
        image, target = dataset[0]
        assert target.shape == (4, 16, 16)
        foreground = target[0] > 0
        assert torch.equal(
            target[3, foreground].long(), (image[0, foreground] > 0.9).long()
        )
        assert (target[3, ~foreground] == -1).all()


@pytest.mark.parametrize(
    "content, message",
    [
        ("{}", "match mask IDs"),
        ('{"2":"A", "9":"B"}', "match mask IDs"),
        ('{"2":"unknown"}', "unknown class"),
        ('{"02":"A"}', "positive integer"),
        ('{"0":"A", "2":"A"}', "positive integer"),
        ('{"2":"A", "2":"B"}', "duplicate"),
        ("[]", "map mask IDs"),
    ],
)
def test_invalid_instance_labels_are_rejected(tmp_path, content, message):
    path = tmp_path / "cell.json"
    path.write_text(content)
    with pytest.raises((ValueError, TypeError), match=message):
        read_labels(path, np.array([[0, 2]], dtype=np.int32), ("A", "B"))


def test_dual_head_preserves_segmentation_and_trains_classifier(tmp_path):
    baseline = CellposeDINO(FakeEncoder())
    path = tmp_path / "model.pt"
    save_weights(path, baseline)
    model = CellposeDINO(FakeEncoder(), num_classes=2)
    with pytest.raises(RuntimeError, match="class_output"):
        load_weights(path, model)
    load_weights(path, model, initialize_classifier=True)
    image = torch.randn(1, 3, 32, 40)
    output = model(image)
    torch.testing.assert_close(output[:, :3], baseline(image), rtol=0, atol=0)
    assert output.shape == (1, 5, 32, 40)
    target = torch.zeros(1, 4, 32, 40)
    target[:, 0] = 1
    target[:, 3, :, 20:] = 1
    loss = CellposeLoss()(output, target)
    loss.backward()
    assert model.class_output.weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0
    assert model.encoder.scale.grad.abs().sum() > 0
    assert model.encoder.patch_embed.proj.weight.grad is None
    model.eval()
    assert run_tiled(model, image, tile_size=32).shape == (1, 5, 32, 40)


def test_class_loss_ignores_background_and_handles_empty_crops():
    prediction = torch.randn(1, 5, 8, 8, requires_grad=True)
    target = torch.zeros(1, 4, 8, 8)
    target[:, 3] = -1
    loss = CellposeLoss()(prediction, target)
    torch.testing.assert_close(loss, CellposeLoss()(prediction[:, :3], target[:, :3]))
    loss.backward()
    assert not prediction.grad[:, 3:].any()
    prediction.grad = None
    target[:, 0, :3, :3] = 1
    target[:, 3, :3, :3] = 1
    CellposeLoss()(prediction, target).backward()
    assert prediction.grad[:, 3:, :3, :3].abs().sum() > 0
    assert not prediction.grad[:, 3:, 3:, :].any()


def test_empty_half_precision_crop_does_not_overflow_class_reduction():
    output = torch.zeros(1, 5, 32, 32, dtype=torch.float16, requires_grad=True)
    with torch.no_grad():
        output[:, 3:] = 100
    target = torch.zeros(1, 4, 32, 32)
    target[:, 3] = -1
    loss = CellposeLoss()(output, target)
    assert torch.isfinite(loss)


def test_prediction_assigns_classes_to_reconstructed_mask_ids():
    masks = np.zeros((32, 32), dtype=np.int32)
    masks[3:9, 3:9] = 2
    masks[21:27, 21:27] = 9
    targets = flow_targets(masks)
    class_logits = np.zeros((2, 32, 32), dtype=np.float32)
    class_logits[0, masks == 2] = 10
    class_logits[1, masks == 9] = 10
    raw = np.concatenate(
        (targets[1:] * FLOW_SCALE, np.where(masks > 0, 10, -10)[None], class_logits)
    ).astype(np.float32)

    class Oracle(torch.nn.Module):
        num_classes = 2

        def forward(self, image):
            return torch.from_numpy(raw).to(image.device)[None]

    model = Oracle()
    result = predict(
        model, masks.astype(np.float32), "cpu", tile_size=32, classes=("A", "B")
    )
    assert len(result.instances) == 2
    for instance in result.instances:
        region = result.masks == instance["mask_id"]
        expected = "A" if np.count_nonzero(region & (masks == 2)) else "B"
        assert instance["class_name"] == expected
        assert instance["score"] > 0.99
    assert model.training
    assert {item["mask_id"] for item in result.instances} == set(
        np.unique(result.masks)
    ) - {0}
    assert classify_instances(np.zeros_like(masks), class_logits, ("A", "B")) == ()


def test_weights_and_resume_reject_reordered_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "build_dinov3", lambda *a, **kw: FakeEncoder())
    cfg = {"model": {"backbone": "vitb16", "patch_stride": 8, "classes": ["A", "B"]}}
    model = build.build_model(cfg, load_backbone=False)
    path = tmp_path / "model.pt"
    save_weights(path, model)
    save_yaml(path.with_suffix(".yaml"), cfg)
    reordered = {"model": {**cfg["model"], "classes": ["B", "A"]}}
    with pytest.raises(ValueError, match="class names/order"):
        build.build_model(reordered, weights=path, load_backbone=False)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, 0, cfg=cfg)
    with pytest.raises(ValueError, match="class names/order"):
        load_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, cfg=reordered)
    path.with_suffix(".yaml").unlink()
    with pytest.raises(FileNotFoundError):
        build.build_model(
            cfg, weights=path, load_backbone=False, initialize_classifier=True
        )
