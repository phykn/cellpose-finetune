import numpy as np
import pytest
import torch

from src.dataset import CellposeDataset, find_pairs, read_array, write_mask


def test_dataset_returns_image_and_flow_target(tmp_path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    masks = np.zeros((32, 32), dtype=np.uint16)
    masks[8:24, 8:24] = 7
    np.save(image_dir / "sample.npy", image)
    np.save(mask_dir / "sample.npy", masks)

    dataset = CellposeDataset(
        image_dir,
        mask_dir,
        crop_size=32,
        augment=False,
    )
    normalized, target = dataset[0]

    assert normalized.shape == (3, 32, 32)
    assert target.shape == (3, 32, 32)
    assert normalized.dtype == torch.float32
    assert target.dtype == torch.float32
    assert torch.equal(target[0] > 0, torch.from_numpy(masks > 0))


def test_find_pairs_reports_missing_stem(tmp_path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    np.save(image_dir / "image.npy", np.zeros((8, 8)))
    np.save(mask_dir / "mask.npy", np.zeros((8, 8)))

    with pytest.raises(ValueError, match="stems do not match"):
        find_pairs(image_dir, mask_dir)


def test_mask_io_round_trip(tmp_path) -> None:
    masks = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path = tmp_path / "masks.tif"

    write_mask(path, masks)

    assert np.array_equal(read_array(path), masks)
