import numpy as np
import pytest

from src.prepare.convert import normalize_image


def test_grayscale_image_is_normalized_without_clipping() -> None:
    image = np.arange(100, dtype=np.float32).reshape(10, 10)

    normalized = normalize_image(image)

    low, high = np.percentile(image, (1.0, 99.0))
    expected = (image - low) / (high - low)
    assert normalized.shape == (3, 10, 10)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[0], expected)
    assert normalized[0, 0, 0] < 0.0
    assert normalized[0, -1, -1] > 1.0
    assert not normalized[1:].any()


def test_rgb_channels_are_normalized_independently() -> None:
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    image = np.stack((values, 10.0 + 2.0 * values, 100.0 - values), axis=-1)

    normalized = normalize_image(image)

    assert normalized.shape == (3, 8, 8)
    np.testing.assert_allclose(normalized[0], normalized[1], atol=1.0e-6)
    np.testing.assert_allclose(normalized[0], normalized[2][::-1, ::-1], atol=1.0e-6)


def test_constant_and_missing_channels_are_zero() -> None:
    image = np.stack(
        (
            np.full((5, 7), 12.0, dtype=np.float32),
            np.arange(35, dtype=np.float32).reshape(5, 7),
        )
    )

    normalized = normalize_image(image, channel_axis=0)

    assert not normalized[0].any()
    assert normalized[1].min() < 0.0
    assert normalized[1].max() > 1.0
    assert not normalized[2].any()


@pytest.mark.parametrize(
    ("image", "channel_axis", "error"),
    (
        (np.zeros((4, 5, 4), dtype=np.uint8), -1, "one and three"),
        (np.zeros((4, 5, 6), dtype=np.uint8), None, "one and three"),
        (np.zeros((2, 3, 4, 5), dtype=np.uint8), None, "spatial dimensions"),
        (np.zeros((4, 5), dtype=np.uint8), 0, "only valid"),
    ),
)
def test_invalid_image_channels_are_rejected(
    image: np.ndarray,
    channel_axis: int | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        normalize_image(image, channel_axis=channel_axis)
