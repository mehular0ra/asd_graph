import os
from typing import Optional, List
import hydra
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.loader import DataLoader

from typing import Tuple

import numpy as np

import matplotlib.pyplot as plt
from collections import Counter



import ipdb


def k_fold_dataloader(cfg: DictConfig,
                         graph_data_list: List,
                         site: np.array) -> List[Tuple[DataLoader, DataLoader]]:

    # n_splits = cfg.training.n_splits  # Number of folds
    n_splits = 10  # Number of folds

    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    dataloaders = []
    for train_index, test_index in skf.split(graph_data_list, site):
        train_data_list = [graph_data_list[i] for i in train_index]
        test_data_list = [graph_data_list[i] for i in test_index]

        # create PyG dataloader
        train_dataloader = DataLoader(
            train_data_list, batch_size=cfg.dataset.batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)

        dataloaders.append((train_dataloader, test_dataloader))

    # add total_steps and steps_per_epoch to cfg
    with open_dict(cfg):
        train_length = ((n_splits-1)/n_splits) * len(graph_data_list)
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    return dataloaders





def init_stratified_dataloader(cfg: DictConfig,
                               graph_data_list: List,
                               site: np.array) -> List[DataLoader]:
    
    train_length = cfg.dataset.train_set * len(graph_data_list)
    # train_length = cfg.dataset.train_set * final_pearson.shape[0]
    train_ratio = cfg.dataset.train_set
    test_ratio = cfg.dataset.test_set

    asd_count = [graph_data_list[i].y.item() == 1.0 for i in range(len(graph_data_list))].count(True)
    print(asd_count)    
    td_count = [graph_data_list[i].y.item() == 0.0 for i in range(len(graph_data_list))].count(True)
    print(td_count)   


    # Stratified split
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, train_size=train_ratio, random_state=42)
    for train_index, test_index in split.split(graph_data_list, site):
        train_data_list = [graph_data_list[i] for i in train_index]
        test_data_list = [graph_data_list[i] for i in test_index]

    # create pyg dataloader
    train_dataloader = DataLoader(
        train_data_list, batch_size=cfg.dataset.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data_list, batch_size=cfg.dataset.batch_size, shuffle=False)

    # add total_steps and steps_per_epoch to cfg
    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    return [train_dataloader, test_dataloader]




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



