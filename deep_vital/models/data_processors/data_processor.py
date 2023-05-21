import math
from typing import Mapping, Optional, Sequence, Union
import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of
from deep_vital.registry import MODELS


@MODELS.register_module()
class DataPreprocessor(BaseDataPreprocessor):
    def __init__(self, pad_size_divisor=1, pad_value: Union[float, int] = 0, non_blocking: Optional[bool] = False):
        super().__init__(non_blocking)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
    
    def forward(self, data: dict, training:bool =False):
        # batch_pad_shape = self._get_pad_shape(data)
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                _batch_input = _batch_input.float()
                batch_inputs.append(_batch_input)
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 3, (
                'The input of `DataPreprocessor` should be a NCL tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            _batch_inputs = _batch_inputs.float()
            l = _batch_inputs.shape[2]
            target_l = math.ceil(
                l / self.pad_size_divisor) * self.pad_size_divisor
            pad_l = target_l - l
            batch_inputs = F.pad(_batch_inputs, (0, pad_l, 0, 0),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}： {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data