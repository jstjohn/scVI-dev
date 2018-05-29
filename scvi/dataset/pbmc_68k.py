import csv

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .dataset import GeneExpressionDataset
import matplotlib.pyplot as plt

key_svmlight = ["CD34+", "CD56+ NK", "CD4+/CD45RA+/CD25- Naive T", "CD4+/CD25 T Reg", "CD8+/CD45RA+ Naive Cytotoxic",
                "CD4+/CD45RO+ Memory", "CD8+ Cytotoxic T", "CD19+ B", "CD4+ T Helper2", "CD14+ Monocyte", "Dendritic"]

key_color_order = ["CD19+ B", "CD14+ Monocyte", "Dendritic", "CD56+ NK", "CD34+", "CD4+/CD25 T Reg",
                   "CD4+/CD45RA+/CD25- Naive T", "CD4+/CD45RO+ Memory", "CD4+ T Helper2",
                   "CD8+/CD45RA+ Naive Cytotoxic",
                   "CD8+ Cytotoxic T"]
colors = ["#1C86EE",  # 1c86ee dodgerblue2
          "#008b00",  # green 4
          "#6A3D9A",  # purple
          "grey",
          "#8B5A2B",  # tan4
          "yellow",
          "#FF7F00",  # orange
          "black",
          "#FB9A99",  # pink
          "#ba55d3",  # orchid
          "red"]

key_names_color = dict(zip(key_color_order, colors))

index_to_color = dict([(i, (k, key_names_color[k])) for i, k in enumerate(key_svmlight)])


class PurePBMC(GeneExpressionDataset):
    def __init__(self, p_sample=1., p_genes=1., to_numpy=False, subset_genes=None):
        sparse_matrix, labels = load_svmlight_file("data/68k/pure_full.svmlight")
        labels = labels - 1
        super(PurePBMC, self).__init__(
            *GeneExpressionDataset.get_attributes_from_list(
                [sparse_matrix[labels == i] for i in range(int(labels[-1] + 1))],
                list_labels=[labels[labels == i] for i in range(int(labels[-1] + 1))]),
            p_sample=p_sample,
            p_genes=p_genes,
            to_numpy=to_numpy,
            subset_genes=subset_genes)


class DonorPBMC(GeneExpressionDataset):
    def __init__(self, p_sample=1., p_genes=1., to_numpy=False, subset_genes=None):
        sparse_matrix, labels = load_svmlight_file("data/68k/68k_assignments.svmlight")
        labels = labels - 1
        # with open('data/68k/pure_use_genes.csv', 'r') as csvfile:
        #     data_reader = csv.reader(csvfile, delimiter='\t')
        #     data_reader.__next__()
        #     pure_use_genes = np.array([int(r[0].split(',')[1]) for r in data_reader])
        # pure_use_genes = pure_use_genes-1 # r indexing
        # nb_genes = sparse_matrix.shape[1]
        # pure_use_genes = pure_use_genes[pure_use_genes < nb_genes]
        # print("Donor dataset to match pure Dataset from %d to %d genes" % (nb_genes, len(pure_use_genes)))
        # sparse_matrix = sparse_matrix[:, pure_use_genes]
        super(DonorPBMC, self).__init__(
            *GeneExpressionDataset.get_attributes_from_matrix(
                sparse_matrix,
                labels=labels),
            p_sample=p_sample,
            p_genes=p_genes,
            to_numpy=to_numpy,
            subset_genes=subset_genes)


def normalize_pca_tsne(donor, n_tsne=10000):
    """
    :param donor: The donor PBMC dataset
    """
    X_new = donor.X[:n_tsne]
    labels_tsne = donor.labels[:n_tsne].ravel()

    # Step 1 - Normalize
    rs = X_new.sum(axis=1)
    rs_med = np.median(rs.A)
    X_new_normalized = (X_new / rs) * rs_med

    # Step 2 - Extract genes
    new_n_genes = 1000
    std_scaler = StandardScaler(with_mean=False)
    std_scaler.fit(X_new_normalized.astype(np.float64))
    subset_genes = np.argsort(std_scaler.var_)[::-1][:new_n_genes]

    X_new_normalized = X_new_normalized[:, subset_genes].A
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_new_normalized)
    X_pca_idx_tsne = TSNE().fit_transform(X_pca)

    plt.figure(figsize=(10, 10))
    for l in range(donor.n_labels):
        plt.scatter(X_pca_idx_tsne[labels_tsne==l, 0], X_pca_idx_tsne[labels_tsne==l, 1],
                    c=index_to_color[l][1],
                   edgecolors='none', s=5)

    filedir = "figures/"+"donor_10000.png"
    print("saving picture at ",filedir)
    plt.savefig(filedir)
