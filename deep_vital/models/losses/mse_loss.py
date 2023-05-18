from typing import Tuple

import torch
from mmengine.model import BaseModule
from torch import nn
import torch.nn.functional as F
from deep_vital.registry import MODELS


@MODELS.register_module()
class MSELoss(BaseModule):
    """Loss function for CAE.

    Compute the align loss and the main loss.

    Args:
        lambd (float): The weight for the align loss.
    """

    def __init__(self, lambd: float=1.0) -> None:
        super().__init__()
        self.lambd = lambd
        self.loss_mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function of CAE Loss.

        Args:
            latent_pred (torch.Tensor): The latent prediction from the
                regressor.
            latent_target (torch.Tensor): The latent target from the teacher
                network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The main loss and align loss.
        """
        # sbp_pred, dbp_pred = pred
        # sbp_target = target[:, 0]
        # dbp_target = target[:, 1]
        # sbp_target.to(torch.float32)
        # sbp_loss = self.loss_mse(sbp_pred, sbp_target.detach())
        
        # pred = torch.cat([pred[0], pred[1]], dim=1)
        # TODO: set float32 in dataset loader
        target = target.to(torch.float32)
        # loss_align = self.loss_mse(pred, target.detach())
        mse_loss = F.mse_loss(pred, target)
        return mse_loss