import torch
import torch.nn.functional as F
from torch import nn

from .. import FLOW_SCALE


class CellposeLoss(nn.Module):
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("prediction must have shape [batch, 3, height, width].")
        if target.shape != prediction.shape:
            raise ValueError("target must have the same shape as prediction.")

        flow_loss = F.mse_loss(prediction[:, :2], FLOW_SCALE * target[:, 1:])
        flow_loss = flow_loss / 2.0
        cell_loss = F.binary_cross_entropy_with_logits(
            prediction[:, 2],
            (target[:, 0] > 0.5).to(prediction.dtype),
        )
        return flow_loss + cell_loss
