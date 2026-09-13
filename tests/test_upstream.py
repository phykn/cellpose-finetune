from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.prepare.flow import masks_to_flows


def test_flows_match_upstream_reference() -> None:
    path = Path(__file__).parent / "fixtures" / "cellpose_a54cb488.npz"
    with np.load(path, allow_pickle=False) as fixture:
        for mask, expected in zip(fixture["masks"], fixture["flows"], strict=True):
            np.testing.assert_allclose(masks_to_flows(mask), expected, atol=1e-6)


def test_pixel_shuffle_matches_upstream_fixed_transposed_convolution() -> None:
    stride = 8
    values = torch.randn(2, 3 * stride**2, 4, 5, requires_grad=True)
    weights = torch.eye(3 * stride**2).reshape(3 * stride**2, 3, stride, stride)
    expected = F.conv_transpose2d(values, weights, stride=stride)
    actual = F.pixel_shuffle(values, stride)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, values, grad)[0],
        torch.autograd.grad(expected, values, grad)[0],
        rtol=0,
        atol=0,
    )
