import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

import ipdb


class HGNNLayer(nn.Module):
	def __init__(self, input_dim, output_dim):
		"""
		One layer of hypergraph neural network

		Args:
			input_dim : number of features of each node in hyergraph
			output_dim : number of output features
		"""
		super(HGNNLayer, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.Linear = nn.Linear(input_dim, output_dim)
    

	def forward(self, x, H):
		"""
		Args:
		x : feature matrix [num_nodes, input_dim]
		H : incidence matrix [num_nodes, num_hyper_edges]
		returns:
		x : output of one layer of hypergraph neural network [num_nodes, output_dim]
		"""
		# compute degree of nodes (D_v)^-0.5
		degree_of_nodes = torch.nan_to_num(torch.pow(torch.diag(
			torch.sum(H, dim=-1)), -0.5), nan=0, posinf=0, neginf=0).to(torch.float32)
		# compute degree of hyper edges (D_e)^-1
		degree_of_edges = torch.nan_to_num(torch.pow(torch.diag(
			torch.sum(H, dim=0)), -1.0), nan=0, posinf=0, neginf=0).to(torch.float32)

		# compute D_v^-0.5 H D_e^-1 H^T D_v^-0.5 x
		x = degree_of_nodes @ x
		x = torch.transpose(H, 0, 1) @ x
		x = degree_of_edges @ x
		x = H @ x
		x = degree_of_nodes @ x

		# apply linear layer
		x = self.Linear(x)
		return x


class HGNN(nn.Module):
	def __init__(self, cfg: DictConfig):
		"""
		Hypergraph neural network containing num_layers HyperNNLayer

		Args:
		input_dim : number of features of each node in hyergraph
		output_dim : number of output features
		hidden_dim : hidden dimension
		num_layers : number of layers
		"""
		super(HGNN, self).__init__()

		self.num_layers = cfg.model.num_layers
		self.dropout = cfg.model.dropout
		self.hidden_dim = cfg.model.hidden_dim
		self.node_feature_dim = cfg.dataset.node_feature_dim
		self.num_nodes = cfg.dataset.num_nodes
		self.batch_size = cfg.training.batch_size

		if self.num_layers > 1:
			self.hnn_layers = [HGNNLayer(self.node_feature_dim, self.hidden_dim)]
			self.hnn_layers += [HGNNLayer(self.hidden_dim, self.hidden_dim)
								for i in range(self.num_layers-1)]
		else:
			self.hnn_layers = [HGNNLayer(self.node_feature_dim, self.hidden_dim)]

		self.hnn_layers = nn.ModuleList(self.hnn_layers)
		self.attention = nn.Linear(self.hidden_dim, 1)  # attention vector
		self.mlp = nn.Linear(self.num_nodes, 1)	

	def forward(self, hgraph):
		"""
		Args:
		hgraph : input hypergraph stored as HyperGraph class
		returns:
		y_hat : logits for each node [num_nodes, output_dim]
		"""
		H = hgraph.incidence_matrix()
		x = hgraph.x.to(torch.float32)

		for j in range(self.num_layers-1):
			x = self.hnn_layers[j](x, H)
			x = F.relu(x)
			x = self.hnn_layers[self.num_layers-1](x, H)

		# Apply learnable attention mechanism
		att_weights = F.softmax(self.attention(x), dim=1)
		x = torch.sum(att_weights * x, dim=1)  # weighted sum of node embeddings

		# Use MLP for classification
		x = x.reshape(self.batch_size, -1)
		y_hat = self.mlp(x)

		return y_hat

