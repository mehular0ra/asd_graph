import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch

from omegaconf import DictConfig

from .DwAttnHGNConv import DwAttnHGNConv

from .Readouts.set_transformer_models import SetTransformer
from .Readouts.janossy_pooling import JanossyPooling

import ipdb


class HeadAggregatorMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = None):
        super(HeadAggregatorMLP, self).__init__()
        if not hidden_size:
            hidden_size = output_size

        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


class DwAttnHGN(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super(DwAttnHGN, self).__init__()
        self.cfg = cfg
        self.num_layers = cfg.model.num_layers
        self.dropout = cfg.model.dropout
        self.hidden_size = cfg.model.hidden_size
        self.num_classes = cfg.dataset.num_classes
        self.node_sz = cfg.dataset.node_sz

        self.num_edges = cfg.dataset.node_sz
        self.heads = cfg.model.heads
        self.concat = cfg.model.concat
        self.attention_mode = cfg.model.attention_mode


        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(DwAttnHGNConv(
                    cfg, i, cfg.dataset.node_feature_sz, self.hidden_size, num_edges=self.num_edges,
                    use_attention=True, attention_mode=self.attention_mode, heads=self.heads, dropout=cfg.model.dropout, concat=self.concat))
            else:
                self.convs.append(DwAttnHGNConv(
                    cfg, i, self.hidden_size, self.hidden_size, num_edges=self.num_edges,
                    use_attention=True, attention_mode=self.attention_mode, heads=self.heads, dropout=cfg.model.dropout, concat=self.concat))

        if self.heads > 1:
            self.aggregator_mlps = torch.nn.ModuleList()
            for i in range(self.num_layers):
                if self.concat:
                    self.aggregator_mlps.append(HeadAggregatorMLP(
                        self.hidden_size * self.heads, self.hidden_size))
                else:
                    self.aggregator_mlps.append(HeadAggregatorMLP(
                        self.hidden_size, self.hidden_size))
                
        if self.cfg.model.readout == 'set_transformer':
            self.readout_layer = SetTransformer(dim_input=self.hidden_size,
                                                num_outputs=1, dim_output=self.hidden_size)
        elif self.cfg.model.readout == 'janossy':
            self.readout_layer = JanossyPooling(
                num_perm=cfg.model.num_perm, in_features=self.hidden_size, fc_out_features=self.hidden_size)
        else:
            self.readout_lin = nn.Linear(
                self.node_sz * self.hidden_size, self.hidden_size)


        self.lin = nn.Linear(self.hidden_size, 1)



    def forward(self, data):
        x, hyperedge_index, hyperedge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        for i in range(self.num_layers):
            ipdb.set_trace()
            # x = self.convs[i](x, hyperedge_index, hyperedge_weight, self.num_edges)
            x = self.convs[i](x, hyperedge_index)
            # Apply the aggregator MLP
            if self.heads > 1:
                x = self.aggregator_mlps[i](x)

            if i < self.num_layers - 1:
                x = F.leaky_relu(x)

        if self.cfg.model.readout in ['set_transformer', 'janossy']:
            x = self.readout_layer(x)
            x = x.squeeze()
        else:
            xs = []
            for graph_idx in batch.unique():
                graph_nodes = x[batch == graph_idx]
                graph_nodes = graph_nodes.view(-1)
                xs.append(self.readout_lin(graph_nodes))
            x = torch.stack(xs).to(x.device)

        x = self.lin(x)
        return x
