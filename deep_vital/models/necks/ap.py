import torch
import torch.nn as nn

from deep_vital.registry import MODELS


@MODELS.register_module()
class AveragePooling(nn.Module):
    def __init__(self, dim=1, kernel_size=2):
        super().__init__()
        assert dim in [1, 2, 3], 'AveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AvgPool1d(kernel_size)
        elif dim == 2:
            self.gap = nn.AvgPool2d((kernel_size, kernel_size))
        else:
            self.gap = nn.AvgPool3d((kernel_size, kernel_size, kernel_size))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs