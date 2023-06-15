import os
from typing import Optional, List
import hydra
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
import torch

import matplotlib.pyplot as plt
from collections import Counter

import logging

import ipdb

def create_graph_data(final_pearson: torch.Tensor,
                      node_feature: torch.Tensor,
                      labels: torch.Tensor,
                      site: torch.Tensor,
                      site_mapping: dict):

    graph_data_list = []
    for i in range(final_pearson.shape[0]):
        edge_index = final_pearson[i].nonzero(as_tuple=False).t().contiguous()

        # create edge weights tensor 'MAKING THEM ABSOLUTE'
        edge_weight = final_pearson[i][edge_index[0], edge_index[1]].clamp(min=0)

        # Check for NaN values in edge_weight tensor
        if torch.isnan(edge_weight).any():
            print(f"Found NaN values in edge weights for graph {i}")

        
        # map site string to integer
        mapped_site = site_mapping[site[i]]

        # Check for NaN values in node_feature tensor
        # if torch.isnan(node_feature).any():
        #     print(f"Found NaN values in node features for graph {i}")

        data = Data(x=node_feature, edge_index=edge_index,
                    y=labels[i], site=mapped_site, edge_weight=edge_weight)
        graph_data_list.append(data)


    return graph_data_list


def init_stratified_dataloader(cfg: DictConfig,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               site: np.array,
                               final_sc: Optional[torch.tensor] = None,
                               data_creation_func=None) -> List[DataLoader]:

    train_length = cfg.dataset.train_set * final_pearson.shape[0]
    train_ratio = cfg.dataset.train_set
    val_ratio = cfg.dataset.val_set
    test_ratio = cfg.dataset.test_set

    # map site names to unique integers
    unique_sites = set(site)
    site_mapping = {name: idx for idx, name in enumerate(unique_sites)}

    total_counts = Counter(site)

    num_graphs, num_nodes = final_pearson.shape[0], final_pearson.shape[1]

    # Create a tensor of indices from 0 to num_nodes
    indices = torch.arange(num_nodes)
    # Create a one-hot encoded tensor
    node_feature = torch.nn.functional.one_hot(indices).float()


    graph_data_list = data_creation_func(
        final_pearson, node_feature, labels, site, site_mapping, final_sc)

    # Stratified split
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio+test_ratio, train_size=train_ratio, random_state=42)
    for train_index, test_valid_index in split.split(final_pearson, site):
        train_data_list = [graph_data_list[i] for i in train_index]
        test_valid_data_list = [graph_data_list[i] for i in test_valid_index]
        site = site[test_valid_index]

    # Relative ratios for second split
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    relative_test_ratio = 1 - relative_val_ratio

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_ratio, train_size=relative_val_ratio, random_state=42)
    for valid_index, test_index in split2.split(test_valid_data_list, site):
        val_data_list = [test_valid_data_list[i] for i in valid_index]
        test_data_list = [test_valid_data_list[i] for i in test_index]


    # create pyg dataloader
    train_dataloader = DataLoader(
        train_data_list, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)

    # add total_steps and steps_per_epoch to cfg
    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # analyze_dataloaders(train_dataloader, val_dataloader, test_dataloader,
    #                     site_mapping, total_counts)
    # analyze_labels(train_dataloader, val_dataloader, test_dataloader)
    return [train_dataloader, val_dataloader, test_dataloader]


def analyze_labels(train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader):
    print("\nAnalyzing labels...")
    # count the number of occurrences of each label in the train set
    train_labels = [label.item()
                    for data in train_dataloader for label in data.y]
    train_counts = Counter(train_labels)

    # count the number of occurrences of each label in the val set
    val_labels = [label.item() for data in val_dataloader for label in data.y]
    val_counts = Counter(val_labels)

    # count the number of occurrences of each label in the test set
    test_labels = [label.item()
                   for data in test_dataloader for label in data.y]
    test_counts = Counter(test_labels)

    # convert counts to percentages
    train_percentages = {
        label: count/len(train_labels)*100 for label, count in train_counts.items()}
    val_percentages = {
        label: count/len(val_labels)*100 for label, count in val_counts.items()}
    test_percentages = {
        label: count/len(test_labels)*100 for label, count in test_counts.items()}

    print("Train counts: ", train_counts)
    print("Val counts: ", val_counts)
    print("Test counts: ", test_counts)

    # plot the data
    categories = ['Train', 'Val', 'Test']
    labels = [0, 1]  # assuming your labels are 0 and 1

    label_0_values = [train_percentages.get(labels[0], 0), val_percentages.get(
        labels[0], 0), test_percentages.get(labels[0], 0)]
    label_1_values = [train_percentages.get(labels[1], 0), val_percentages.get(
        labels[1], 0), test_percentages.get(labels[1], 0)]

    x = np.arange(len(categories))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, label_0_values,
                    width, label=f'Label {labels[0]}')
    rects2 = ax.bar(x + width/2, label_1_values,
                    width, label=f'Label {labels[1]}')

    ax.set_xlabel('Datasets')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of each label in each dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
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


def analyze_dataloaders(train_dataloader: DataLoader, val_dataloader: DataLoader, test_dataloader: DataLoader, site_mapping: dict, total_counts: Counter):
    print("\nAnalyzing dataloaders...")
    # count the number of occurrences of each site in the train set
    train_sites = [site.item()
                   for data in train_dataloader for site in data.site]
    train_counts = Counter(train_sites)

    # count the number of occurrences of each site in the val set
    val_sites = [site.item()
                 for data in val_dataloader for site in data.site]
    val_counts = Counter(val_sites)

    # count the number of occurrences of each site in the test set
    test_sites = [site.item()
                  for data in test_dataloader for site in data.site]
    test_counts = Counter(test_sites)

    # reverse the site_mapping dictionary for easier plotting
    reverse_site_mapping = {v: k for k, v in site_mapping.items()}

    # convert counts to percentages of the total number of each site's subjects
    train_percentages = {reverse_site_mapping[site]: (
        count / total_counts[reverse_site_mapping[site]])*100 for site, count in train_counts.items()}
    val_percentages = {reverse_site_mapping[site]: (
        count / total_counts[reverse_site_mapping[site]])*100 for site, count in val_counts.items()}
    test_percentages = {reverse_site_mapping[site]: (
        count / total_counts[reverse_site_mapping[site]])*100 for site, count in test_counts.items()}

    print("Train percentages: ", train_percentages)
    print("Val percentages: ", val_percentages)
    print("Test percentages: ", test_percentages)

    # plot the data
    categories = ['Train', 'Val', 'Test']
    labels = list(reverse_site_mapping.values())

    label_values = {category: [percentages.get(label, 0) for label in labels] for category, percentages in zip(
        categories, [train_percentages, val_percentages, test_percentages])}

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, label_values['Train'], width, label='Train')
    rects2 = ax.bar(x, label_values['Val'], width, label='Val')
    rects3 = ax.bar(x + width, label_values['Test'], width, label='Test')

    ax.set_xlabel('Sites')
    ax.set_ylabel('Percentage')
    ax.set_title(
        'Percentage of subjects from each site in train, val and test sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # Save the plot in the specified directory
    original_dir = hydra.utils.get_original_cwd()
    img_dir = os.path.join(original_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(f"{img_dir}/site_percentage_plot.png")
    plt.close(fig)



