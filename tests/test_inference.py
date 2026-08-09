import numpy as np
import pytest
import torch
from torch import nn

from src import inference
from src.inference import Prediction, predict, run_tiled


class PointwiseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)


@pytest.mark.parametrize(
    ("shape", "tile_size"),
    (
        ((7, 11), 16),
        ((35, 53), 32),
        ((129, 173), 64),
    ),
)
def test_tiled_pointwise_model_matches_full_image_for_any_shape(
    shape: tuple[int, int],
    tile_size: int,
) -> None:
    torch.manual_seed(0)
    model = PointwiseModel()
    image = torch.randn(1, 3, *shape)

    expected = model(image)
    actual = run_tiled(model, image, tile_size=tile_size, overlap=0.25)

    assert actual.shape == (1, 3, *shape)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5)


def test_tiled_inference_preserves_input_and_parameter_gradients() -> None:
    model = PointwiseModel()
    image = torch.randn(1, 3, 41, 57, requires_grad=True)

    output = run_tiled(model, image, tile_size=32, overlap=0.25)
    output.square().mean().backward()

    assert image.grad is not None
    assert image.grad.shape == image.shape
    assert bool(torch.isfinite(image.grad).all())
    assert model.conv.weight.grad is not None
    assert bool(torch.isfinite(model.conv.weight.grad).all())


@pytest.mark.parametrize("tile_size", (8, 18, 385, True))
def test_invalid_tile_sizes_are_rejected(tile_size: object) -> None:
    image = torch.zeros(1, 3, 16, 16)

    with pytest.raises(ValueError, match="tile_size"):
        run_tiled(PointwiseModel(), image, tile_size=tile_size)  # type: ignore[arg-type]


def test_predict_returns_original_shape_for_an_empty_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward_states = []

    class EmptyModel(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            forward_states.append((self.training, torch.is_inference_mode_enabled()))
            output = torch.zeros(
                image.shape[0],
                3,
                *image.shape[-2:],
                device=image.device,
                dtype=image.dtype,
            )
            output[:, 2].fill_(-10.0)
            return output

    calls = {}
    real_compute_masks = inference.compute_masks

    def recording_compute_masks(
        flow_logits: np.ndarray,
        cell_logits: np.ndarray,
        **kwargs: object,
    ) -> np.ndarray:
        calls["flow_shape"] = flow_logits.shape
        calls["cell_shape"] = cell_logits.shape
        calls.update(kwargs)
        return real_compute_masks(flow_logits, cell_logits, **kwargs)

    monkeypatch.setattr(inference, "compute_masks", recording_compute_masks)
    image = np.full((23, 37), 7, dtype=np.uint16)
    model = EmptyModel()

    result = predict(
        model,
        image,
        device="cpu",
        tile_size=32,
        overlap=0.25,
        niter=17,
        cellprob_threshold=0.2,
        flow_threshold=None,
        min_size=3,
        max_size_fraction=0.3,
    )

    assert isinstance(result, Prediction)
    assert result.masks.shape == image.shape
    assert np.issubdtype(result.masks.dtype, np.integer)
    assert not result.masks.any()
    assert result.flow_logits.shape == (2, *image.shape)
    assert result.flow_logits.dtype == np.float32
    assert result.cell_logits.shape == image.shape
    assert result.cell_logits.dtype == np.float32
    assert np.allclose(result.cell_logits, -10.0)
    assert forward_states
    assert all(state == (False, True) for state in forward_states)
    assert model.training
    assert calls == {
        "flow_shape": (2, *image.shape),
        "cell_shape": image.shape,
        "niter": 17,
        "cellprob_threshold": 0.2,
        "flow_threshold": None,
        "min_size": 3,
        "max_size_fraction": 0.3,
        "device": torch.device("cpu"),
    }
