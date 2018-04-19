import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

plt.switch_backend('agg')


def show_t_sne(latent, labels, title):
    if latent.shape[1] != 2:
        latent = TSNE().fit_transform(latent)
    plt.figure(figsize=(10, 10))
    plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap=plt.get_cmap("tab20", 15), edgecolors='none')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if not os.path.exists('clusters'):
        os.mkdir('clusters')
    plt.savefig('clusters/{}'.format(title))
    print("saving tsne figure as {}.png".format(title))
    plt.savefig("{}".format(title))
