from typing import Optional
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine import MessageHub
from deep_vital.registry import METRICS

message_hub = MessageHub.get_current_instance()


@METRICS.register_module()
class MAE(BaseMetric):
    metric = 'MAE'
    default_prefix = 'MAE'

    def __init__(self,
                 gt_key='gt_label',
                 pred_key='pred_label',
                 loss_key=None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.gt_key = gt_key
        self.pred_key = pred_key
        self.loss_key = loss_key

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            result = {
                'pred_label': data_sample[self.pred_key],
                'gt_label': data_sample[self.gt_key]
            }
            # if self.loss_key:
            #     result['pred_loss'] = data_sample[self.loss_key]
            self.results.append(result)

    def compute_metrics(self, results):
        metrics = {}
        target = torch.stack([res['gt_label'] for res in results])
        pred = torch.stack([res['pred_label'] for res in results])
        if self.loss_key:
            # loss = torch.stack([res['pred_loss'] for res in results])
            # val_loss = loss.mean()
            # metrics['pred_loss'] = val_loss
            sbp_pred = pred[:, 0]
            dbp_pred = pred[:, 1]
            sbp_target = target[:, 0]
            dbp_target = target[:, 1]
            sbp_mse_loss = F.mse_loss(sbp_pred, sbp_target.detach())
            dbp_mse_loss = F.mse_loss(dbp_pred, dbp_target.detach())
            metrics['pred_loss'] = sbp_mse_loss + dbp_mse_loss

        diff = abs(pred - target)
        sbp_mean = diff[:, 0].mean()
        dbp_mean = diff[:, 1].mean()
        metrics['sbp_mae'] = sbp_mean
        metrics['dbp_mae'] = dbp_mean
        return metrics
