from omegaconf import DictConfig, open_dict
from sklearn.model_selection import StratifiedShuffleSplit

from .fc import load_data
from .construct_hypergraph import HyperGraph
from .mini_batch import create_mini_batch

from typing import List
import torch.utils as utils
import logging

import ipdb


def stratified_split(cfg, data_list, labels, sites):
    train_ratio = cfg.dataset.train_set
    test_ratio = cfg.dataset.test_set

    # Stratified split
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, train_size=train_ratio, random_state=42)
    for train_index, test_index in split.split(data_list, sites):
        train_list = [data_list[i] for i in train_index]
        test_list = [data_list[i] for i in test_index]

    return train_list, test_list


def dataset_factory(cfg: DictConfig):
    labels, site, fc, sc, t1 = load_data(cfg)
    hypergraphs = [HyperGraph(fc[i] if fc is not None else None,
                              sc[i] if sc is not None else None,
                              t1[i] if t1 is not None else None,
                              fc[i] if fc is not None else None,
                              labels[i],
                              cfg.dataset.k) for i in range(len(labels))]

    data_list = [hypergraphs[i] for i in range(len(labels))]
    train_list, test_list = stratified_split(cfg, data_list, labels, site)
    return train_list, test_list

    # Create mini-batches
        

