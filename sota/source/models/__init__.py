from omegaconf import DictConfig

from .HGNN import HGNN


def model_factory(config: DictConfig):
    return eval(config.model.name)(config)
