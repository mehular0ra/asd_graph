import numpy as np
import torch

from omegaconf import DictConfig


def top_percent_edges(matrix, percent):
    """
    Keep the top percent of edges by absolute value.
    """
    flattened = np.sort(np.abs(matrix.flatten()))

    index = int((1.0 - percent/100.0) * len(flattened))

    threshold = flattened[index]

    mask = np.abs(matrix) >= threshold
    matrix_masked = matrix * mask
    return matrix_masked


def load_data(cfg: DictConfig):

    if cfg.dataset.fc:
        fc_data = np.load(cfg.dataset.fc_path, allow_pickle=True).item()
        fc = fc_data["corr"]
        labels = fc_data["label"]
        site = fc_data['site']

        for i in range(fc.shape[0]):
            fc[i] = top_percent_edges(fc[i], cfg.dataset.perc_edges)

        fc = torch.from_numpy(fc).float()
    else:
        fc = None

    if cfg.dataset.sc:
        sc_data = np.load(cfg.dataset.sc_path, allow_pickle=True).item()
        sc = sc_data["sc"]
        labels = sc_data["label"]
        site = sc_data['site']

        for i in range(sc.shape[0]):
            sc[i] = top_percent_edges(sc[i], cfg.dataset.perc_edges)

        sc = torch.from_numpy(sc).float()
    else:
        sc = None

    if cfg.dataset.t1:
        t1_data = np.load(cfg.dataset.t1_path, allow_pickle=True).item()
        t1 = t1_data["t1"]
        labels = t1_data["label"]
        site = t1_data['site']

        t1 = torch.from_numpy(t1).float()
    else:
        t1 = None

    labels = torch.from_numpy(labels).float()
    return labels, site, fc, sc, t1
