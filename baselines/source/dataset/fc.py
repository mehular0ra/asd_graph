import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.signal import hilbert

import ipdb


def normalized_covariance_matrix(data):
    """
    Calculate the normalized covariance matrix for time series data.   
    Args:
        data (ndarray): A 2D numpy array containing the time series data.

    Returns:
        ndarray: The normalized covariance matrix.
    """
    # Calculate the covariance matrix
    data = hilbert(data.T).T  # apply hilbert transform
    cov_matrix = np.cov(data, rowvar=False)

    # Calculate the diagonal matrix of standard deviations
    std_devs = np.sqrt(np.diag(cov_matrix))
    std_dev_matrix = np.diag(std_devs)

    # Calculate the inverse of the standard deviation matrix
    inv_std_dev_matrix = np.linalg.pinv(std_dev_matrix)

    # Calculate the normalized covariance matrix
    normalized_cov_matrix = np.dot(
        inv_std_dev_matrix, np.dot(cov_matrix, inv_std_dev_matrix))

    return normalized_cov_matrix


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


def load_fc_data(cfg: DictConfig):
    if cfg.dataset.name == 'fc_abide1':
        fc_data = np.load(cfg.dataset.fc_path, allow_pickle=True)
        final_pearson = np.array(fc_data["corr"])
        labels = np.array(fc_data["label"])
        site = np.array(fc_data['site'])
        # labels which are 2 are converted to 0
        labels[labels == 2] = 0
    else:
        fc_data = np.load(cfg.dataset.fc_path, allow_pickle=True).item()
        final_pearson = fc_data["corr"]
        labels = fc_data["label"]
        site = fc_data['site']


    transform = OmegaConf.select(cfg, "dataset.transform")
    if transform == 'hilbert':
        timeseries = fc_data['timeseries']
        fc_cov = []
        for i in range(timeseries.shape[0]):
            fc_cov.append(normalized_covariance_matrix(
                np.transpose(timeseries[i])))

        final_pearson = np.imag(np.array(fc_cov))

    # # Apply edge pruning
    # for i in range(final_pearson.shape[0]):
    #     final_pearson[i] = top_percent_edges(final_pearson[i], cfg.dataset.perc_edges)

    final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
    return final_pearson, labels, site
