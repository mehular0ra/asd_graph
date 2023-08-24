import torch
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter, add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch import Tensor
from typing import Optional

import ipdb




class HypergraphConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_edges: int, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=False)

        # Define hyperedge_weight as a learnable Parameter
        self.hyperedge_weight = Parameter(torch.Tensor(num_edges))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        # glorot(self.hyperedge_weight)  # Initializing hyperedge weights
        # Initializing hyperedge weights
        torch.nn.init.normal_(self.hyperedge_weight, mean=1.0, std=0.1)

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_edges: Optional[int] = None) -> Tensor:
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        x = self.lin(x)

        D = scatter(self.hyperedge_weight[hyperedge_index[1]],
                    hyperedge_index[0], dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)),
                    hyperedge_index[1], dim=0, dim_size=num_edges, reduce='sum')
        # B = torch.zeros(num_edges, device=x.device)
        # B.scatter_add_(0, hyperedge_index[1], x.new_ones(hyperedge_index.size(1)))
        B = 1.0 / B
        B[B == float("inf")] = 0
        ipdb.set_trace()
        # Multiply B with the learnable hyperedge_weight
        B = B * self.hyperedge_weight

        out = self.propagate(hyperedge_index, x=x, norm=B,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip(
            [0]), x=out, norm=D, size=(num_edges, num_nodes))

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor) -> Tensor:
        return norm_i.view(-1, 1) * x_j
