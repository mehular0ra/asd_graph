import numpy as np
from sklearn.neighbors import KDTree
import torch

import ipdb

""" HyperGraph structure
            for a mini-batch it will store a big hypergraph
            representing the entire batch
        Args:
            x: node features  [num_nodes x num_feats]
            y: hypergraph labels   [num_hyper_graphs]
            hyper_edge_index: list of hyperedges [2, k] where k is defined in
            the above cell.
        """


class HyperGraph(object):
    def __init__(self, fc, sc, t1, x, y, k=5, device='cuda:0'):
        if fc is None and sc is None and t1 is None:
            raise ValueError(
                "At least one of 'fc', 'sc', or 't1' must be not None.")
        
        self.device = device


        self.fc = fc if fc is not None else None
        self.sc = sc if sc is not None else None
        self.t1 = t1 if t1 is not None else None
        self.k = k
        self.x = x.to(torch.float32)
        self.y = y
        self.num_nodes = self.x.shape[0]

        self.hyper_edge_index = self.create_hyper_edges()


    def to(self, device):
        """Move all tensors to the given device."""
        for name, value in self.__dict__.items():
                if torch.is_tensor(value):
                    setattr(self, name, value.to(device))
        return self


    def create_hyper_edges_from_matrix(self, matrix):
        hyper_edge_index = torch.zeros(
            [2, matrix.shape[0] * (self.k+1)], dtype=torch.long)
        for node in range(matrix.shape[0]):
            # Get k closest nodes (include the node itself)
            connected_nodes = np.argpartition(
                matrix[node, :], -self.k-1)[-self.k-1:]
            # Assign each node its own hyperedge ID and associate its nearest neighbors
            for idx, connected_node in enumerate(connected_nodes):
                # Hyper edge's node
                hyper_edge_index[0, node*(self.k+1)+idx] = connected_node
                # Hyperedge ID
                hyper_edge_index[1, node*(self.k+1)+idx] = node

        return hyper_edge_index


    def create_hyper_edges(self):
        hyper_edge_index = []
        offset = 0  # Initialize offset
        if self.fc is not None:
            fc_hyper_edges = self.create_hyper_edges_from_matrix(self.fc)
            hyper_edge_index.append(fc_hyper_edges)
            offset += fc_hyper_edges.shape[1]  # Update offset

        if self.sc is not None:
            sc_hyper_edges = self.create_hyper_edges_from_matrix(self.sc)
            # Add offset to hyperedge IDs
            sc_hyper_edges[1] += offset
            hyper_edge_index.append(sc_hyper_edges)
            offset += sc_hyper_edges.shape[1]  # Update offset

        if self.t1 is not None:
            tree = KDTree(self.t1)
            indices_list = []
            for node in range(self.t1.shape[0]):
                _, indices = tree.query(self.t1[node, :].reshape(1, -1), k=self.k+1)
                # Add offset to hyperedge IDs
                indices_list.extend([(idx, node+offset) for idx in indices[0]])
            hyper_edge_index.append(torch.tensor(indices_list, dtype=torch.long).t().contiguous())

        self.num_hyper_edges = offset  # Update total number of hyperedges
        hyper_edge_index =  torch.cat(hyper_edge_index, dim=1)
        return hyper_edge_index


    def incidence_matrix(self):
        hyper_edge_index_cpu = self.hyper_edge_index.to('cpu')
        return torch.sparse.LongTensor(hyper_edge_index_cpu,
                                    torch.ones(
                                        (hyper_edge_index_cpu.shape[1],)),
                                    torch.Size((self.num_nodes, self.num_hyper_edges))).to_dense().to(self.device)
