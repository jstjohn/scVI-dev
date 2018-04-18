import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


# CLUSTERING METRICS
def entropy_batch_mixing(latent_space, batches, max_number=500):
    # latent space: numpy matrix of size (number_of_cells, latent_space_dimension)
    # with the encoding of the different inputs in the latent space
    # batches: numpy vector with the batch indices of the cells
    n_samples = len(latent_space)
    keep_idx = np.random.choice(np.arange(n_samples), size=min(len(latent_space), max_number), replace=False)
    latent_space, batches = latent_space[keep_idx], batches[keep_idx]

    def entropy(hist_data):
        n_batches = len(np.unique(hist_data))
        if n_batches > 2:
            raise ValueError("Should be only two clusters for this metric")
        frequency = np.mean(hist_data == 1)
        if frequency == 0 or frequency == 1:
            return 0
        return -frequency * np.log(frequency) - (1 - frequency) * np.log(1 - frequency)

    n_samples = latent_space.shape[0]
    distance = np.zeros((n_samples, n_samples))
    neighbors_graph = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            distance[i, j] = distance[j, i] = sum((latent_space[i] - latent_space[j]) ** 2)

    for i, d in enumerate(distance):
        neighbors_graph[i, d.argsort()[:51]] = 1
    kmatrix = neighbors_graph - np.identity(latent_space.shape[0])

    score = 0
    for t in range(50):
        indices = np.random.choice(np.arange(latent_space.shape[0]), size=100)
        score += np.mean([entropy(
            batches[kmatrix[indices].nonzero()[1][kmatrix[indices].nonzero()[0] == i]]
        ) for i in range(100)])
    return score / 50


def histogram_of_clusters(vae, data_loader, filename="clustering.png"):
    thetas = []
    cell_types = []
    for i_batch, (sample_batch, _, _, _, cell_types_batch) in enumerate(data_loader):
        sample_batch = Variable(sample_batch)
        if vae.using_cuda:
            sample_batch = sample_batch.cuda()
        theta_batch = vae.get_theta(sample_batch)
        thetas += [theta_batch]
        cell_types += [cell_types_batch]
    thetas = torch.cat(thetas)
    cell_types = torch.cat(cell_types)
    clusters = thetas.max(1)[1].data.cpu()  # size is latent space
    cell_types_names = data_loader.dataset.cell_types
    n_cell_types = len(cell_types_names)
    clustering_output = np.array([[
        (cell_types[clusters == i] == j).sum() for i in range(vae.n_latent)
    ] for j in range(n_cell_types)])

    levels = np.concatenate([np.zeros((1, vae.n_latent)), clustering_output.cumsum(axis=0)], axis=0)
    ind = np.arange(vae.n_latent)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    bars = [plt.bar(ind, levels[i + 1], width, bottom=levels[i], color=c)[0] for i, c in enumerate(colors)]

    plt.ylabel('Number of cells')
    plt.title('Clustering with ProdLDA')
    plt.xticks(ind, ('G%d' % i for i in range(vae.n_latent)))
    plt.legend(bars, cell_types_names)
    plt.savefig('data/output/' + filename)
    #plt.show()
