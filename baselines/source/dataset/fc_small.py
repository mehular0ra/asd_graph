import numpy as np
import torch
from omegaconf import DictConfig, open_dict


def load_fc_small_data(cfg: DictConfig):

    fc_data = np.load(cfg.dataset.fc_path, allow_pickle=True).item()

    final_pearson = fc_data["corr"]
    labels = fc_data["label"]
    site = fc_data['site']

    final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]

    return final_pearson, labels, site
