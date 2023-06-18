from omegaconf import DictConfig
from .GCN import GCN
from .SignedGCN import SignedGCN



def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config)
