import torch
import torch.nn.functional as F
from torch import nn

from .. import FLOW_SCALE


class CellposeLoss(nn.Module):
    def __init__(self, class_loss_weight: float = 1.0) -> None:
        super().__init__()
        self.class_loss_weight = class_loss_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.ndim != 4 or prediction.shape[1] < 3:
            raise ValueError(
                "prediction must have shape [batch, 3 + classes, height, width]."
            )
        num_classes = prediction.shape[1] - 3
        expected = (prediction.shape[0], 4 if num_classes else 3, *prediction.shape[2:])
        if target.shape != expected:
            raise ValueError(f"target must have shape {expected}.")

        flow_loss = F.mse_loss(prediction[:, :2], FLOW_SCALE * target[:, 1:3])
        flow_loss = flow_loss / 2.0
        cell_loss = F.binary_cross_entropy_with_logits(
            prediction[:, 2],
            (target[:, 0] > 0.5).to(prediction.dtype),
        )
        loss = flow_loss + cell_loss
        if num_classes:
            labels = target[:, 3].long()
            valid = (target[:, 0] > 0.5) & (labels >= 0)
            class_logits = prediction[:, 3:].permute(0, 2, 3, 1)
            # Empty crops must have finite loss and zero classification gradients.
            class_loss = (
                F.cross_entropy(class_logits[valid].float(), labels[valid])
                if valid.any()
                else class_logits.float().sum() * 0.0
            )
            loss = loss + self.class_loss_weight * class_loss
        return loss
