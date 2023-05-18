from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn

from mmengine.model import BaseModel
from deep_vital.registry import MODELS
from deep_vital.structures import DataSample


@MODELS.register_module()
class BPDenseHead(BaseModel):
    def __init__(self, loss, init_cfg = None):
        super().__init__(init_cfg)
        
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        
        # output layer
        self.SBP = nn.Linear(2048 * 9, 1)
        self.DBP = nn.Linear(2048 * 9, 1)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pre_logits = self.pre_logits(feats)

        x = pre_logits.flatten(1)  # without batch dim

        x_sbp = self.SBP(x)
        x_dbp = self.DBP(x)
        return x_sbp, x_dbp

    def predict(self, x, batch_data_samples, rescale):
        return self.forward(x)

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs):
        # The part can be traced by torch.fx
        pred_val = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(pred_val, data_samples, **kwargs)
        return losses

    def _get_loss(self, pred_val: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.stack([i.gt_label for i in data_samples])

        sbp_pred, dbp_pred = pred_val
        sbp_target = target[:, 0][..., None]
        dbp_target = target[:, 1][..., None]
        # compute loss
        losses = dict()
        # loss = self.loss_module(pred_val, target)
        sbp_loss = self.loss_module(sbp_pred, sbp_target)
        dbp_loss = self.loss_module(dbp_pred, dbp_target)
        losses['sbp_loss'] = sbp_loss
        losses['dbp_loss'] = dbp_loss
        losses['loss'] = sbp_loss + dbp_loss
        return losses