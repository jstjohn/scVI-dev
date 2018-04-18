# -*- coding: utf-8 -*-

"""Handling datasets.
For the moment, is initialized with a torch Tensor of size (n_cells, nb_genes)"""
import numpy as np
import scipy.sparse as sp_sparse
import torch
from torch.utils.data import Dataset


class GeneExpressionDataset(Dataset):
    """Gene Expression dataset. It deals with:
    - log_variational expression -> torch.log(1 + X)
    - local library size normalization (mean, var) per batch
    """

    def __init__(self, Xs, list_labels=None, gene_names=None):
        # Args:
        # Xs: a list of numpy tensors with .shape[1] identical (total_size*nb_genes)
        # or a list of scipy CSR sparse matrix,
        # or transposed CSC sparse matrix (the argument sparse must then be set to true)
        self.nb_genes = Xs[0].shape[1]
        self.n_batches = len(Xs)

        assert all(X.shape[1] == self.nb_genes for X in Xs), "All tensors must have same size"

        new_Xs = []
        local_means = []
        local_vars = []
        batch_indices = []
        for i, X in enumerate(Xs):
            log_counts = torch.log(torch.from_numpy(np.asarray((X.sum(axis=1)), dtype=float))
                                   .type(torch.FloatTensor))
            local_mean = torch.mean(log_counts)
            local_var = torch.var(log_counts)
            new_Xs += [X]
            local_means += [local_mean * torch.ones((X.shape[0], 1))]
            local_vars += [local_var * torch.ones((X.shape[0], 1))]
            batch_indices += [torch.LongTensor(i * np.ones((X.shape[0], 1)))]

        self.local_means = torch.cat(local_means, dim=0)
        self.local_vars = torch.cat(local_vars, dim=0)
        self.batch_indices = torch.cat(batch_indices)

        self.X = sp_sparse.vstack(new_Xs).astype(np.float32)
        self.total_size = self.X.shape[0]

        # This will be changed by the .h5 file format
        all_labels = []
        if list_labels is not None:
            for labels in list_labels:
                all_labels += [torch.LongTensor(labels.reshape(-1, 1))]
            self.all_labels = torch.cat(all_labels, dim=0)
        else:
            self.all_labels = torch.zeros_like(self.batch_indices)  # We might want default label values

        if gene_names is not None:
            self.gene_names = np.char.upper(gene_names)  # Take an upper case convention for gene names

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        return self.X[idx].toarray()[0], self.local_means[idx], self.local_vars[idx], \
               self.batch_indices[idx], self.all_labels[idx]

    @staticmethod
    def train_test_split(*Xs, train_size=0.75):
        """
        A substitute for the sklearn function to avoid the dependency
        """
        Xs = [np.array(X) for X in Xs]
        split_idx = int(train_size * len(Xs[0]))
        all_indices = np.arange(len(Xs[0]))
        train_indices = np.random.choice(all_indices, size=split_idx, replace=False)
        test_indices = np.array(list(set(all_indices).difference(set(train_indices))))
        split_list = [[X[train_indices], X[test_indices]] for X in Xs]
        return [X for Xs in split_list for X in Xs]
