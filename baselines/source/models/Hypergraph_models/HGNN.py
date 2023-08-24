import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.conv import HypergraphConv


class HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, use_attention=False):
        super(HGNN, self).__init__()
        self.conv1 = HypergraphConv(
            in_channels, hidden_size, use_attention=use_attention)
        self.conv2 = HypergraphConv(
            hidden_size, hidden_size, use_attention=use_attention)
        self.classifier = Linear(hidden_size, 1)

    def convert_to_edge_list(self, H, edge_weights):
        node_indices, hyperedge_indices = torch.nonzero(H, as_tuple=True)
        hyperedge_index = torch.stack([node_indices, hyperedge_indices])
        # hyperedge_weight = edge_weights[node_indices, hyperedge_indices]
        return hyperedge_index, 

    def forward(self, data):
        x, H, edge_weights = data.x, data.H, data.edge_weights
        hyperedge_index = self.convert_to_edge_list(H, edge_weights)

        x = self.conv1(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        out = self.classifier(x)
        return out
