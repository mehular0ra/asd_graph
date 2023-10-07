from omegaconf import DictConfig, open_dict
import os
import torch
from .fc import load_fc_data
from .dataloader import init_stratified_dataloader, k_fold_dataloader

from typing import List
import torch.utils as utils
import logging

from .construct_graph import create_graph_data
from .construct_hyperaph import create_hypergraph_data

import ipdb


def create_dataloader(cfg: DictConfig) -> List[utils.data.DataLoader]:
    """create dataloader from dataset

    Args:
        cfg (DictConfig): config

    Returns:
        List[utils.data.DataLoader]: dataloader
    """
    # create dataset
    datasets = load_fc_data(cfg)

    # graph data creation
    data_creation = cfg.model.get("data_creation", "graph")
    data_creation_func = "create_" + data_creation + "_data"
    graph_data_list, site = eval(data_creation_func)(cfg, *datasets)

    # dataloader creation
    if cfg.kfold:
        dataloaders = k_fold_dataloader(cfg, graph_data_list, site)
    else:
        dataloaders = init_stratified_dataloader(cfg, graph_data_list, site)

    return dataloaders


def dataset_factory(cfg: DictConfig, k=None) -> List[utils.data.DataLoader]:

    logging.info('cfg.dataset.name: %s', cfg.dataset.name)

    if k is not None:
        logging.info('kfold: %s', k)

        # add total_steps and steps_per_epoch to cfg
        n_splits = 10  # Number of folds
        num_subjects = cfg.dataset.num_subjects
        with open_dict(cfg):
            train_length = ((n_splits-1)/n_splits) * num_subjects
            # total_steps, steps_per_epoch for lr schedular
            cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs


        data_creation = cfg.model.get("data_creation", "graph")
        # check if kfold dataset directory exists (check if a directory exists)
        data_path = cfg.dataset.kfold_dataset_dir + data_creation + '/'
        if os.path.isdir(data_path):
            logging.info('kfold dataset directory exists')
        else:
            logging.info('creating kfold dataset directory')
            os.mkdir(data_path)
            dataloaders = create_dataloader(cfg)
            for i, (train_dataloader, test_dataloader) in enumerate(dataloaders):
                torch.save(train_dataloader, data_path + f'train_fold_{i}.pt')
                torch.save(test_dataloader, data_path + f'test_fold_{i}.pt')

        train_dataloader = torch.load(data_path + f'train_fold_{k}.pt')
        test_dataloader = torch.load(data_path + f'test_fold_{k}.pt')
        return [train_dataloader, test_dataloader]

    return create_dataloader(cfg)