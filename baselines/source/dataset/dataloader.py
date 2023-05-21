import os
from typing import Optional, List
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import DataLoader, Data

import numpy as np
import torch

import ipdb

import matplotlib.pyplot as plt
from collections import Counter

def create_graph_data(final_pearson: torch.Tensor,
                      node_feature: torch.Tensor,
                      labels: torch.Tensor,
                      site: torch.Tensor,
                      site_mapping: dict):

    graph_data_list = []
    print(final_pearson.shape[0])
    for i in range(final_pearson.shape[0]):
        edge_index = final_pearson[i].nonzero(as_tuple=False).t().contiguous()

        # map site string to integer
        mapped_site = site_mapping[site[i]]
        data = Data(x=node_feature, edge_index=edge_index,
                    y=labels[i], site=mapped_site)
        graph_data_list.append(data)

    return graph_data_list


def init_stratified_dataloader(cfg: DictConfig,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               site: np.array,
                               final_sc: Optional[torch.tensor] = None) -> List[DataLoader]:

    # map site names to unique integers
    unique_sites = set(site)
    site_mapping = {name: idx for idx, name in enumerate(unique_sites)}

    total_counts = Counter(site)

    num_graphs, num_nodes = final_pearson.shape[0], final_pearson.shape[1]
    node_feature = torch.ones((num_graphs, num_nodes))

    graph_data_list = create_graph_data(
        final_pearson, node_feature, labels, site, site_mapping)

    # Stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(final_pearson, site):
        train_data_list = [graph_data_list[i] for i in train_index]
        test_data_list = [graph_data_list[i] for i in test_index]

    # create pyg dataloader
    train_dataloader = DataLoader(
        train_data_list, batch_size=cfg.dataset.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)

    # analyze dataloaders/ plots
    analyze_dataloaders(train_dataloader, test_dataloader,
                        site_mapping, total_counts)
    analyze_labels(train_dataloader, test_dataloader)

    return [train_dataloader, test_dataloader]


def analyze_labels(train_dataloader: DataLoader, test_dataloader: DataLoader):
    # count the number of occurrences of each label in the train set
    train_labels = [label.item()
                    for data in train_dataloader for label in data.y]
    train_counts = Counter(train_labels)

    # count the number of occurrences of each label in the test set
    test_labels = [label.item()
                   for data in test_dataloader for label in data.y]
    test_counts = Counter(test_labels)

    # convert counts to percentages
    train_percentages = {
        label: count/len(train_labels)*100 for label, count in train_counts.items()}
    test_percentages = {
        label: count/len(test_labels)*100 for label, count in test_counts.items()}

    print("Train percentages: ", train_percentages)
    print("Test percentages: ", test_percentages)

    # plot the data
    labels = [0, 1]  # if your labels are 0 and 1
    train_values = [train_percentages.get(label, 0) for label in labels]
    test_values = [test_percentages.get(label, 0) for label in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, train_values, width, label='Train')
    rects2 = ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage by label')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the plot in the specified directory
    # This will be the directory where your application was launched
    original_dir = hydra.utils.get_original_cwd()
    img_dir = os.path.join(original_dir, 'imgs')
    # This will create the directory if it doesn't already exist
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(f"{img_dir}/label_percentage_plot.png")
    plt.close(fig)





def analyze_dataloaders(train_dataloader: DataLoader, test_dataloader: DataLoader, site_mapping: dict, total_counts: Counter):

    # count the number of occurrences of each site in the train set
    train_sites = [site.item()
                   for data in train_dataloader for site in data.site]
    train_counts = Counter(train_sites)

    # count the number of occurrences of each site in the test set
    test_sites = [site.item()
                  for data in test_dataloader for site in data.site]
    test_counts = Counter(test_sites)

    # reverse the site_mapping dictionary for easier plotting
    reverse_site_mapping = {v: k for k, v in site_mapping.items()}

    # convert counts to percentages of the total number of each site's subjects
    train_percentages = {reverse_site_mapping[site]: (
        count / total_counts[reverse_site_mapping[site]])*100 for site, count in train_counts.items()}
    test_percentages = {reverse_site_mapping[site]: (
        count / total_counts[reverse_site_mapping[site]])*100 for site, count in test_counts.items()}

    print("Train percentages: ", train_percentages)
    print("Test percentages: ", test_percentages)

    # plot the data
    labels = list(reverse_site_mapping.values())
    # use 0 if site doesn't exist in train set
    train_values = [train_percentages.get(label, 0) for label in labels]
    # use 0 if site doesn't exist in test set
    test_values = [test_percentages.get(label, 0) for label in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, train_values, width, label='Train')
    rects2 = ax.bar(x + width/2, test_values, width, label='Test')

    ax.set_xlabel('Sites')
    ax.set_ylabel('Percentage')
    ax.set_title(
        'Percentage of subjects from each site in train and test sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the plot in the specified directory
    original_dir = hydra.utils.get_original_cwd()  # This will be the directory where your application was launched
    img_dir = os.path.join(original_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)  # This will create the directory if it doesn't already exist
    plt.savefig(f"{img_dir}/site_percentage_plot.png")
    plt.close(fig)


