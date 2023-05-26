from omegaconf import DictConfig
from .GCN import GCN
from .GAT import GAT
from .GraphSAGE import GraphSAGE




def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config)
