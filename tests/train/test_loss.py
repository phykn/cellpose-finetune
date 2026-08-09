import torch
import torch.nn.functional as F

from src import FLOW_SCALE
from src.train.loss import CellposeLoss


def test_cellpose_loss_matches_reference_formula() -> None:
    prediction = torch.tensor(
        [[[[1.0]], [[-2.0]], [[0.5]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[[1.0]], [[0.25]], [[-0.5]]]])

    loss = CellposeLoss()(prediction, target)
    expected = F.mse_loss(prediction[:, :2], FLOW_SCALE * target[:, 1:]) / 2.0
    expected = expected + F.binary_cross_entropy_with_logits(
        prediction[:, 2],
        target[:, 0],
    )
    loss.backward()

    assert torch.allclose(loss, expected)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_cellpose_loss_uses_background_flow_pixels() -> None:
    prediction = torch.zeros(1, 3, 2, 2)
    target = torch.zeros_like(prediction)
    target[:, 1, 0, 0] = 1.0

    loss = CellposeLoss()(prediction, target)
    cell_loss = F.binary_cross_entropy_with_logits(
        prediction[:, 2],
        target[:, 0],
    )

    assert loss > cell_loss
