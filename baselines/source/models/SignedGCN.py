import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, EdgePooling
from torch_geometric.data import Batch

from omegaconf import DictConfig

from .EdgeWeightedSignedConv import EdgeWeightedSignedConv

import ipdb


class SignedGCN(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SignedGCN, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(EdgeWeightedSignedConv(
                    cfg.dataset.node_feature_sz, self.hidden_size // 2))
            else:
                self.convs.append(EdgeWeightedSignedConv(
                    self.hidden_size, self.hidden_size // 2))

        self.readout_lin = nn.Linear(self.node_sz * self.hidden_size, self.hidden_size)

        self.lin = nn.Linear(self.hidden_size, 1)

        # TODO: Add different pooling methods

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        # 1. Obtain node embeddings
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)

            if torch.isnan(x).any():
                print(f"Found NaN values in output tensor in layer {i}")

        # 2. Readout layer
        xs = []
        for graph_idx in batch.unique():
            graph_nodes = x[batch == graph_idx]
            graph_nodes = graph_nodes.view(-1)
            xs.append(self.readout_lin(graph_nodes))
        x = torch.stack(xs).to(x.device)
        # x = global_max_pool(x, batch)
        # x = torch.stack([self.lincomb(x[i:i+self.node_sz].flatten())
        #                 for i in range(0, x.shape[0], self.node_sz)]).to('cuda')

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
