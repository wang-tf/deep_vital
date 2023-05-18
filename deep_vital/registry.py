from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import Registry

HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['deep_vital.engine.hooks'])

MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['deep_vital.models'])
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['deep_vital.datasets'])
METRICS = Registry('metric', parent=MMENGINE_METRICS, locations=['deep_vital.evaluation'])
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS, locations=['deep_vital.datasets'])
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS, locations=['deep_vital.visualization'])
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS, locations=['deep_vital.visualization'])
