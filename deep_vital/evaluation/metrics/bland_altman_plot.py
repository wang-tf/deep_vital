from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
from mmengine.evaluator import BaseMetric
from mmengine.visualization import Visualizer
from mmengine import MessageHub
from deep_vital.registry import METRICS

message_hub = MessageHub.get_current_instance()


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1.numpy())
    data2 = np.asarray(data2.numpy())
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.cla()
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes('RGBA', (w, h), buf.tostring())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    plt.close()
    return rgb_image


@METRICS.register_module()
class BlandAltmanPlot(BaseMetric):
    metric = 'BlandAltmanPlot'

    def __init__(self,
                 gt_key='gt_label',
                 pred_key='pred_label',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.gt_key = gt_key
        self.pred_key = pred_key
        self.visualizer = Visualizer.get_current_instance()

    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            result = {
                'pred_label': data_sample[self.pred_key],
                'gt_label': data_sample[self.gt_key],
            }
            self.results.append(result)

    def compute_metrics(self, results):
        metrics = {}
        target = torch.stack([res['gt_label'] for res in results])
        pred = torch.stack([res['pred_label'] for res in results])

        sbp_BAP = bland_altman_plot(pred[:, 0], target[:, 0])
        dbp_BAP = bland_altman_plot(pred[:, 1], target[:, 1])
        current_step = message_hub.get_info('epoch')
        self.visualizer.add_image('sbp_BAP', sbp_BAP, current_step)
        self.visualizer.add_image('dbp_BAP', dbp_BAP, current_step)
        return metrics
