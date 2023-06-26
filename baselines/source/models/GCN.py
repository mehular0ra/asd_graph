import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

from omegaconf import DictConfig
import ipdb


class GCN(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(GCN, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                self.convs.append(GCNConv(cfg.dataset.node_feature_sz, self.hidden_size))
            else:
                self.convs.append(GCNConv(self.hidden_size,self.hidden_size))

        self.lincomb = nn.Linear(
            self.node_sz * self.hidden_size, self.hidden_size)
                
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(),
            # nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//4, 1)
        )
                
        ### TODO: Add different pooling methods

    def convert_edge_positive(self, edge_index, edge_weight):
        edge_index = edge_index[:, edge_weight > 0]
        edge_weight = edge_weight[edge_weight > 0]
        return edge_index, edge_weight

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        edge_index, edge_weight = self.convert_edge_positive(edge_index, edge_weight)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)

            if torch.isnan(x).any():
                print(f"Found NaN values in output tensor in layer {i}")

        # x = global_mean_pool(x, batch)
        x = torch.stack([self.lincomb(x[i:i+self.node_sz].flatten())
                            for i in range(0, x.shape[0], self.node_sz)]).to('cuda')

        x = self.fc(x)

        return x
