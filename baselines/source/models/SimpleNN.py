import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

import ipdb

class SimpleNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        print("SimpleNN init")
        super(SimpleNN, self).__init__()
        node_feature = cfg.dataset.node_feature_sz
        hidden_dim = cfg.model.hidden_dim
        output_dim = cfg.dataset.num_classes
        self.fc1 = nn.Linear(node_feature*node_feature, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, pearson):

        x = F.relu(self.fc1(pearson))
        output = self.fc2(x)
        return output
