import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
plt.style.use('seaborn-white')


def get_data(epoch, base_location, train=True):
    tsne_folder = "train_tsne" if train else "test_tsne"

    # Define the file paths using the base location
    x_file_paths = [
        os.path.join(base_location, tsne_folder, f"x_epoch_{epoch}_iter_{i}.npy") for i in range(6)
    ]

    label_file_paths = [
        os.path.join(base_location, tsne_folder, f"label_epoch_{epoch}_iter_{i}.npy") for i in range(6)
    ]

    # Load the x files
    x_arrays = [np.load(file_path) for file_path in x_file_paths]
    # Load the label files
    label_arrays = [np.load(file_path) for file_path in label_file_paths]

    # Concatenate arrays along the first axis
    x_final = np.concatenate(x_arrays, axis=0)
    label_final = np.concatenate(label_arrays, axis=0)

    return x_final, label_final


def plot_data(x_final, label_final, base_location, epoch, train=True):
    tsne = TSNE(angle=0.2, init='pca', method='barnes_hut',
                metric='euclidean', n_iter=2000, perplexity=50, random_state=42)
    x_tsne = tsne.fit_transform(x_final)

    # Plotting
    plt.figure(figsize=(7, 6), dpi=150)
    plt.style.use('seaborn-white')

    label_classes = [0, 1]
    colors = ['blue', 'red']
    labels = ["TD", "ASD"]

    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(x_tsne[label_final == i, 0], x_tsne[label_final == i, 1], marker='o',
                    color=c, label=label, edgecolor='w', s=50, alpha=0.8)

    # Determine the title text based on the base location
    if base_location.endswith("22-08-45"):
        title_text = 'HyperGEL'
        print("title_text: ", title_text)
    elif base_location.endswith("22-09-04"):
        title_text = 'HyperGraphGCN'
    elif base_location.endswith("22-09-15"):
        title_text = 'GCN'

    # Customize plot attributes
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    leg = plt.legend(loc='best', fontsize=10, frameon=True, edgecolor='black')
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    # Save the plot
    directory_name = os.path.basename('.')
    train_test_folder = "train" if train else "test"
    save_directory = os.path.join(
        '/home/mehul/asd_graph/Notebooks', f"{directory_name}_plots", train_test_folder, title_text)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.axis('off')
    plt.savefig(f"{save_directory}/epoch_{epoch}.png",
                bbox_inches='tight', dpi=150)
    plt.close()

    return save_directory


def get_data_and_plot(epoch, base_location, train=True):
    x_final, label_final = get_data(epoch, base_location, train=train)
    save_directory = plot_data(
        x_final, label_final, base_location, epoch, train=train)
    return save_directory


base_locations = [
    "/home/mehul/asd_graph/baselines/outputs/2023-09-05/22-08-45",
    "/home/mehul/asd_graph/baselines/outputs/2023-09-05/22-09-04",
    "/home/mehul/asd_graph/baselines/outputs/2023-09-05/22-09-15"
]

for base_location in base_locations:
    print("base_location: ", base_location)
    for epoch in range(100):  # For epochs 0-99
        save_directory = get_data_and_plot(epoch, base_location, train=True)
        save_directory = get_data_and_plot(epoch, base_location, train=False)
        print(f"Saved epoch {epoch} to {save_directory}\n")
    print(f"Saved epoch {epoch} to {save_directory}")
    print("\n#####################\n")

# # hypergraphgcn
# base_location = "/home/mehul/asd_graph/baselines/outputs/2023-09-05/22-09-04"
# for epoch in range(100):  # For epochs 0-99
#     save_directory = get_data_and_plot(epoch, base_location, train=True)
