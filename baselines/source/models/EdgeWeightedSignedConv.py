from torch import Tensor
import torch
from torch_geometric.nn.conv import SignedConv
from torch.nn import Linear

import ipdb


class EdgeWeightedSignedConv(SignedConv):
    def __init__(self, in_channels, out_channels, first_aggr=True):
        super(EdgeWeightedSignedConv, self).__init__(
            in_channels, out_channels, first_aggr)

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        pos_edge_index = edge_index[:, edge_weight > 0]
        neg_edge_index = edge_index[:, edge_weight < 0]

        pos_edge_weight = edge_weight[edge_weight > 0]
        neg_edge_weight = edge_weight[edge_weight < 0]

        if isinstance(x, Tensor):
            x = (x, x)

        # `propagate` calls `message` internally
        out_pos = self.propagate(pos_edge_index, x=x,
                                 edge_weight=pos_edge_weight, size=None)
        out_neg = self.propagate(neg_edge_index, x=x,
                                 edge_weight=neg_edge_weight, size=None)

        if self.first_aggr:
            out_pos = self.lin_l(out_pos)
            out_neg = self.lin_r(out_neg)
        else:
            out_pos = self.lin_r(out_pos)
            out_neg = self.lin_l(out_neg)

        # # Calculate the weights/proportions
        # num_edges = edge_weight.size(0)
        # num_pos_edges = (edge_weight > 0).sum().item()
        # num_neg_edges = num_edges - num_pos_edges
        # pos_weight = num_pos_edges / num_edges
        # neg_weight = num_neg_edges / num_edges

        # # Scale the outputs
        # out_pos = out_pos * pos_weight
        # out_neg = out_neg * neg_weight

        # # Combine them
        # out = out_pos + out_neg
        # return out

        return torch.cat([out_pos, out_neg], dim=-1)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j
