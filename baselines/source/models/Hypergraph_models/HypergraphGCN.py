import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, HypergraphConv
from torch_geometric.data import Batch

from omegaconf import DictConfig


class HypergraphGCN(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(HypergraphGCN, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(HypergraphConv(
                    cfg.dataset.node_feature_sz, self.hidden_size))
            else:
                self.convs.append(HypergraphConv(
                    self.hidden_size, self.hidden_size))

        self.readout_lin = nn.Linear(
            self.node_sz * self.hidden_size, self.hidden_size)

        self.lin = nn.Linear(self.hidden_size, 1)

    def forward(self, data):
        x, hyperedge_index, hyperedge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, hyperedge_index, hyperedge_weight)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)

        xs = []
        for graph_idx in batch.unique():
            graph_nodes = x[batch == graph_idx]
            graph_nodes = graph_nodes.view(-1)
            xs.append(self.readout_lin(graph_nodes))
        x = torch.stack(xs).to(x.device)

        x = self.lin(x)

        return x
