import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

# from torch_geometric.experimental import disable_dynamic_shapes
from .experimental import disable_dynamic_shapes

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

import ipdb


class DwAttnHGNConv(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.3,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        ### learn the hyperedge weights ###
        self.num_edges = kwargs['num_edges']
        self.learned_he_weights = Parameter(torch.Tensor(self.num_edges))
        ### ###

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    ### hyperedge weight function ###
    def _enforce_non_negativity(self, grad):
        """
        This hook function enforces non-negativity constraint on the 
        learned_he_weights tensor during the backward pass.
        """
        self.learned_he_weights.data = torch.nn.functional.relu(
            self.learned_he_weights.data)
        return grad

    @ disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor,
                hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:

        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)


        # x = self.lin(x)

        alpha = None
        if self.use_attention:

            if hyperedge_attr is None:
                # Change 'sum' to 'mean', 'max', etc
                node_features_for_hyperedges = x[hyperedge_index[0]]
                hyperedge_attr = scatter(node_features_for_hyperedges,
                                         hyperedge_index[1], dim=0, reduce='sum')

            # assert hyperedge_attr is not None
            x = self.lin(x)
            x = x.view(-1, self.heads, self.out_channels)

            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1,
                                                 self.heads, self.out_channels)

            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=x.size(0))
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        ### Multiply hyperedge_weight with B ###
        # Multiply hyperedge_weight with B
        BATCH_SIZE = hyperedge_weight.shape[0] // self.learned_he_weights.shape[0]
        replicated_weights = self.learned_he_weights.repeat(BATCH_SIZE)
        hyperedge_weight = hyperedge_weight * replicated_weights

        modified_hyperedge_weight = hyperedge_weight * B
        ### ###

        modified_hyperedge_weight = hyperedge_weight * B
        out = self.propagate(hyperedge_index, x=x, norm=modified_hyperedge_weight, alpha=alpha,
                             size=(num_nodes, num_edges))

        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
