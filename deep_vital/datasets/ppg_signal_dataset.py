import os
import h5py
import numpy as np
from mmengine.dataset import BaseDataset
from deep_vital.registry import DATASETS, TRANSFORMS


def expanduser(path):
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    if isinstance(path, (str, os.PathLike)):
        return os.path.expanduser(path)
    else:
        return path


@DATASETS.register_module()
class RppgData(BaseDataset):
    def __init__(self,
                 ann_file,
                 used_subjects=(),
                 metainfo=None,
                 data_root='',
                 data_prefix='',
                 filter_cfg=None,
                 indices=None,
                 serialize_data=True,
                 test_mode=False,
                 pipeline=(),
                 lazy_init=False,
                 max_refetch=1000):
        self.label = None
        self.rppg = None
        self.subject_idx = None
        self.used_subjects=used_subjects
        self.num = None

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
        
        ann_file = expanduser(ann_file)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(ann_file=ann_file,
                         metainfo=metainfo,
                         data_root=data_root,
                         data_prefix=data_prefix,
                         filter_cfg=filter_cfg,
                         indices=indices,
                         serialize_data=serialize_data,
                         pipeline=transforms,
                         test_mode=test_mode,
                         lazy_init=lazy_init,
                         max_refetch=max_refetch)

    def load_data_list(self):
        assert os.path.exists(self.ann_file), self.ann_file

        data = h5py.File(self.ann_file, 'r')
        self.label = np.array(data.get('/label')).T
        self.rppg = np.array(data.get('/rppg')).T
        self.subject_idx = np.array(data.get('/subject_idx'), dtype=int)[0, :]
        subjects_list = np.unique(self.subject_idx)

        if self.used_subjects:
            idx_used = np.where(np.isin(self.subject_idx, self.used_subjects))[-1]
            self.label = self.label[idx_used]
            self.rppg = self.rppg[idx_used]
            self.subject_idx = self.subject_idx[idx_used]
        
        self.num = self.subject_idx.shape[0]
        
        data_list = []
        for _label, _rppg, _subject_idx in zip(self.label, self.rppg, self.subject_idx):
            # sbp_label = _label[0]
            # dbp_label = _label[1]
            # _label = _label.reshape((1, 2, 1))
            info = {'gt_label': _label, 'rppg': _rppg, 'subject_idx': _subject_idx}
            data_list.append(info)
        return data_list
