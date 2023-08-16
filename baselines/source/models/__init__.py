from omegaconf import DictConfig
from .GCN import GCN
from .GraphSAGE import GraphSAGE
from .SignedGCN import SignedGCN
from .GAT import GAT



def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config)
