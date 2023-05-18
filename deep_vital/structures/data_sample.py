from typing import List, Sequence, Union
import numpy as np
import torch
from mmengine.structures import BaseDataElement
from mmengine.utils import is_str

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]


def format_label(value) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    # assert value.ndim == 1, \
    #     f'The dims of value should be 1, but got {value.ndim}.'

    return value


class DataSample(BaseDataElement):
    def set_gt_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self
