from typing import Optional
import torch
from mmengine.evaluator import BaseMetric
from mmengine import MessageHub
from deep_vital.registry import METRICS

message_hub = MessageHub.get_current_instance()


@METRICS.register_module()
class MAE(BaseMetric):
    metric = 'MAE'

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
            if self.loss_key:
                result['pred_loss'] = data_sample[self.loss_key]
            self.results.append(result)

    def compute_metrics(self, results):
        metrics = {}
        target = torch.stack([res['gt_label'] for res in results])
        pred = torch.stack([res['pred_label'] for res in results])
        if self.loss_key:
            loss = torch.stack([res['pred_loss'] for res in results])
            val_loss = loss.mean()
            metrics['val_loss'] = val_loss

        diff = abs(pred - target)
        sbp_mean = diff[:, 0].mean()
        dbp_mean = diff[:, 1].mean()
        metrics['sbp_mae'] = sbp_mean
        metrics['dbp_mae'] = dbp_mean
        return metrics
