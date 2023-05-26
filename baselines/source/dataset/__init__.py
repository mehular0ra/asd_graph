from omegaconf import DictConfig, open_dict
from .fc_small import load_fc_small_data
from .dataloader import init_stratified_dataloader

from typing import List
import torch.utils as utils
import logging


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    logging.info('cfg.dataset.name: %s', cfg.dataset.name)

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)
    
    dataloaders = init_stratified_dataloader(cfg, *datasets) 
    
    return dataloaders
