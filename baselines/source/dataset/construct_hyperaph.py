"""
transform graphs (represented by edge list) to hypergraph (represented by node_dict & edge_dict)
"""
from typing import Optional
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics.pairwise import cosine_distances as cos_dis
import torch

import ipdb





def construct_H_with_KNN(X, K_neig, is_probH=False):
    """
    Initialize hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: 2D array representing the pairwise distances between nodes
    :param K_neig: The number of neighbor expansion
    :param is_probH: Prob Vertex-Edge matrix or binary
    :return: N_nodes x N_hyperedge
    """
    n_nodes = X.shape[0]

    # Construct hypergraph incidence matrix H
    H = np.zeros((n_nodes, n_nodes))

    for center_idx in range(n_nodes):
        dis_vec = X[center_idx]

        nearest_idx = np.argsort(np.abs(dis_vec))[-K_neig:]

        for node_idx in nearest_idx:
            if is_probH:
                H[node_idx, center_idx] = np.abs(
                    dis_vec[node_idx]) / torch.sum(np.abs(dis_vec[nearest_idx]))
                
                # Softmax normalization
                # H[node_idx, center_idx] = softmax(np.abs(dis_vec))[node_idx]

                # Inverse distance weighting
                # H[node_idx, center_idx] = 1 / np.abs(dis_vec[node_idx])
            else:
                H[node_idx, center_idx] = 1.0

    return H

def generate_G_from_H(H):
    """
    Calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]

    # The weight of the hyperedge
    W = np.ones(n_edge)

    # The degree of the node
    DV = np.sum(H * W, axis=1)

    # The degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.diag(np.power(DE, -1))
    DV2 = np.diag(np.power(DV, -0.5))
    W = np.diag(W)
    H = np.matrix(H)
    HT = H.T

    G = DV2 @ H @ W @ invDE @ HT @ DV2
    return G

def create_hypergraph_data(cfg: DictConfig,
                           final_pearson: torch.Tensor,
                           node_feature: torch.Tensor,
                           labels: torch.Tensor,
                           site: torch.Tensor,
                           site_mapping: dict,
                           final_sc: Optional[torch.Tensor] = None):

    K_neigs = cfg.model.K_neigs
    is_probH = cfg.model.is_probH

    H_list = []
    G_list = []
    for subject_matrix in final_pearson:
        H_tmp = construct_H_with_KNN(subject_matrix, K_neigs, is_probH)
        G_tmp = generate_G_from_H(H_tmp)
        H_list.append(H_tmp)
        G_list.append(G_tmp)
    return G_list
