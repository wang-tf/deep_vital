from typing import List
from abc import ABCMeta, abstractmethod
import torch
from torch import nn
from mmengine.model import BaseModel
from deep_vital.registry import MODELS
from deep_vital.structures import DataSample


@MODELS.register_module()
class BPResNet1D(BaseModel):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None) -> None:
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.head = MODELS.build(head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        feats = self.extract_feat(batch_inputs)

        results_list = self.head.predict(feats,
                                         batch_data_samples,
                                         rescale=rescale)
        val_loss = self.head.loss(feats, batch_data_samples)
        results_list = torch.cat([results_list[0], results_list[1]], dim=1)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list, val_loss)
        return batch_data_samples

    def _forward(self, batch_inputs, batch_data_samples=None):
        x = self.extract_feat(batch_inputs)
        results = self.head.forward(x)
        return results

    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, inputs, data_samples, mode='tensor'):
        if mode == 'tensor':
            feats = self._forward(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        
        return self.head.loss(feats, data_samples)

    def add_pred_to_datasample(self, data_samples, results_list, loss_list=None):
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_label in zip(data_samples, results_list):
            data_sample.pred_label = pred_label
        if loss_list is not None:
            for data_sample, pred_loss in zip(data_sample, loss_list):
                data_sample.pred_loss = pred_loss
        return data_samples