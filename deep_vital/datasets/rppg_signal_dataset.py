
from deep_vital.registry import DATASETS
from .ppg_signal_dataset import PpgData


@DATASETS.register_module()
class RppgData(PpgData):
    _signal_name = 'rppg'