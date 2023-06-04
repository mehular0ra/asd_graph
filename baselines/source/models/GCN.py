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

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(),
            # nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//4, self.num_classes)
        )

        # Initialize the list of convolutional layers
        self.convs = torch.nn.ModuleList()

        if self.num_layers < 2:
            # TODO: try setting add_self_loops: False
            self.convs.append(GCNConv(cfg.dataset.node_feature_sz, self.hidden_size))  
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
                          self.hidden_size))
        
        ### TODO: Add different pooling methods



    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

 
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.leaky_relu(x)

            if torch.isnan(x).any():
                print(f"Found NaN values in output tensor in layer {i}")


        x = self.convs[-1](x, edge_index, edge_weight)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x

    

# check if the model works
if __name__ == "__main__":
    # print model
    model = GCN()
    print(model)
