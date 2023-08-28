from omegaconf import DictConfig
from .GCN import GCN
from .GraphSAGE import GraphSAGE
from .SignedGCN import SignedGCN
from .GAT import GAT
from .Hypergraph_models.HGNN import HGNN
from .Hypergraph_models.HypergraphGCN import HypergraphGCN
from .Hypergraph_models.DwHGN import DwHGN
from .Hypergraph_models.AttnHGN import AttnHGN
from .Hypergraph_models.DwAttnHGN import DwAttnHGN



def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config)
