import numpy as np
import pytest

from src.data.image import read_array, write_mask


def test_png_rejects_truncating_instance_labels(tmp_path):
    mask = np.array([[0, 65535, 65536]], dtype=np.uint32)
    with pytest.raises(ValueError, match="65535"):
        write_mask(tmp_path / "mask.png", mask)
    assert not (tmp_path / "mask.png").exists()
    for suffix in (".npy", ".tiff"):
        path = tmp_path / f"mask{suffix}"
        write_mask(path, mask)
        np.testing.assert_array_equal(read_array(path), mask)


def test_png_preserves_supported_labels(tmp_path):
    mask = np.array([[0, 65535]], dtype=np.uint32)
    path = tmp_path / "mask.png"
    write_mask(path, mask)
    np.testing.assert_array_equal(read_array(path), mask)
