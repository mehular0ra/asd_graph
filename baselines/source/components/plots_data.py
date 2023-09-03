import numpy as np




def tsne_plot_data(x, labels, epoch, iteration):
    x_numpy = x.detach().cpu().numpy()
    labels_numpy = labels.cpu().numpy()
    np.save(f'x_epoch_{epoch}_iter_{iteration}.npy', x_numpy)
    np.save(
        f'labels_epoch_{epoch}_iter_{iteration}.npy', labels_numpy)
