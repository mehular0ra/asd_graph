import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

from omegaconf import DictConfig


class GCN(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(GCN, self).__init__()
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes

        # Initialize the list of convolutional layers
        self.convs = torch.nn.ModuleList()

        if self.num_layers < 2:
            self.convs.append(GCNConv(cfg.dataset.node_feature_sz, self.num_classes))
            return
            
        # Add the first layer (input layer)
        self.convs.append(
            GCNConv(cfg.dataset.node_feature_sz, self.hidden_size))

        # Add the hidden layers
        # -2 because we manually add the first and the last layers
        for _ in range(self.num_layers - 2):
            self.convs.append(GCNConv(self.hidden_size,
                              self.hidden_size))

        # Add the last layer (output layer)
        self.convs.append(GCNConv(self.hidden_size,
                          self.num_classes))
        

        ### TODO: Add different pooling methods
        ### TODO: Add below code: 
        # self.fc = nn.Sequential(
        #     nn.Linear(8 * sizes[-1], 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 2)
        # )

    def forward(self, x, edge_index):
        # For each layer in the network...
        for i in range(self.num_layers - 1):
            # Apply the layer, then activation function (ReLU), then dropout
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # For the last layer, just apply the layer (no activation or dropout)
        x = self.convs[-1](x, edge_index)

        x = self.custom_mean_pool(x)
        # x = global_mean_pool(x, Batch.batch)

        return x

    def custom_mean_pool(self, x):
        batch_size = x.shape[0] // 400
        x = x.view(batch_size, 400, -1).mean(dim=1)
        return x
