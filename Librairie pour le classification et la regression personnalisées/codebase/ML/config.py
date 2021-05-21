from dataclasses import dataclass, asdict

import yaml
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import make_scorer

from .cross_validation import CROSS_VALIDATION
from .feature_selection import FEATURES_SELECTOR
from .model import MODELS, combine_models
from .sampling import SAMPLERS
from .data_normalization import SCALERS
from .scoring import METRICS
from .utils import *


@dataclass
class Dataset:
    target_col: str
    filepath: str
    drop_col: List = None



@dataclass
class Config:
    target_col: str
    filepath: str
    task: str
    experiment: str
    config_dir: str
    cross_validation: Union[List, Dict]
    feature_selection: Union[List, Dict] = None
    metrics: Union[List, str, Dict] = None
    models: Union[List, str, Dict] = '--all'
    scale: Union[List, str, Dict] = None
    samplers: List = None
    drop_col: List = None
    seed: int = 42
    shuffle: bool = True
    enable_ensemble_methods: bool = False
    use_class_weight: bool = False

    def __post_init__(self):
        self.dataset = Dataset(drop_col=self.drop_col, target_col=self.target_col, filepath=self.filepath)
        experiment, seed, dataset, config_dir, metrics, models, cross_val, feature_selection, scales, samplers = \
            self.experiment, self.seed, self.dataset, self.config_dir, self.metrics, self.models, \
            self.cross_validation, \
            self.feature_selection, self.scale, self.samplers
        task= self.task
        self.config_file = '%s/%s_config.yaml' % (config_dir, experiment)
        self.models = create_instances(models, MODELS[task])
        if self.enable_ensemble_methods:
            self.models += combine_models(self.models, MODELS[task])
        self.cross_validation = create_instances(cross_val, CROSS_VALIDATION)
        self.samplers = create_instances(samplers, SAMPLERS)
        self.scale = create_instances(scales, SCALERS)
        self.metrics = create_instances(metrics, METRICS[task], __call__= make_scorer)
        self.feature_selection = create_instances(feature_selection, FEATURES_SELECTOR, __call__= SelectKBest)

    def save(self, *args, **kwargs):
        with open(self.config_file, 'w') as f:
            yaml.dump(asdict(self), f)
