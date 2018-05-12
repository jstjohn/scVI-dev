import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scvi.utils import no_grad, eval_modules, to_cuda

plt.switch_backend('agg')


def show_t_sne(latent, labels, n_samples=1000):
    idx_t_sne = np.random.permutation(len(latent))[:n_samples]
    if latent.shape[1] != 2:
        latent = TSNE().fit_transform(latent[idx_t_sne])
    plt.figure(figsize=(10, 10))
    plt.scatter(latent[:, 0], latent[:, 1], c=(np.array(labels)[idx_t_sne]).ravel(), edgecolors='none')
    plt.axis("off")
    plt.tight_layout()
    filedir = "figures/"
    filepath = filedir + 'tsne.png'
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    plt.savefig(filepath)
    print("saving tsne figure as %s" % filepath)


@no_grad()
@eval_modules()
def plot_latent(vae, data_loader):
    n = vae.n_latent_layers
    fig, axes = plt.subplots(n, 1, sharey=True, figsize=(5, 5 * n))
    latents = [[]] * n
    labels = []
    for tensorlist in data_loader:
        if vae.use_cuda:
            tensorlist = to_cuda(tensorlist)
        sample_batch, local_l_mean, local_l_var, batch_index, label = tensorlist
        latents_ = vae.get_latent(sample_batch, label)  # might include batch_index afterwards
        latents = [l + [l_] for l, l_ in zip(latents, latents_)]
        labels += [label]
    labels = np.array(torch.cat(labels)).ravel()
    latents = [np.array(torch.cat(l)) for l in latents]
    pca = PCA(n_components=2)
    latents2d = [pca.fit_transform(l) for l in latents]
    if isinstance(axes, plt.Axes):
        axes = [axes]
    for i, ax in enumerate(axes[::-1]):
        latent = latents2d[i]
        ax.scatter(latent[:, 0], latent[:, 1], c=labels, edgecolors='none')
        ax.set_title("Latent Layer %d" % i)

    plt.tight_layout()
    filedir = "figures/"
    filepath = filedir + 'pca-latent.png'
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    plt.savefig(filepath)
    print("Figure saved at %s" % filepath)
