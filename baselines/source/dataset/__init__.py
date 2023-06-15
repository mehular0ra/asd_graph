from omegaconf import DictConfig, open_dict
from .fc_small import load_fc_small_data
from .fc import load_fc_data
from .dataloader import init_stratified_dataloader, create_graph_data, create_hypergraph_data

from typing import List
import torch.utils as utils
import logging


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    logging.info('cfg.dataset.name: %s', cfg.dataset.name)

    datasets = load_fc_data(cfg)


    data_creation = cfg.model.get("data_creation", "graph")
    data_creation_func = "create_" + data_creation + "_data"

   # Use eval to call the function named data_creation_func
    try:
        dataloaders = eval(data_creation_func)(cfg, *datasets)
    except NameError:
        raise ValueError(
            f"{data_creation_func} is not a valid function. Please check the function name in your config file.")
    return dataloaders
