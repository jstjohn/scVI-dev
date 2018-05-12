#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests for `scvi` package."""
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from run_benchmarks import run_benchmarks
from scvi.benchmark import run_benchmarks_classification
from scvi.dataset import SyntheticDataset
from scvi.metrics.log_likelihood import compute_true_log_likelihood
from scvi.models import VAEC, VAE, SVAEC, LVAEC, LVAE, DVAE
from scvi.train import train, train_semi_supervised


def test_synthetic_1():
    run_benchmarks("synthetic", n_epochs=1, use_batches=True, model=VAE, tt_split=0.1)
    run_benchmarks_classification("synthetic", n_epochs=1)


def test_synthetic_2():
    run_benchmarks("synthetic", n_epochs=1, model=SVAEC, benchmark=True, tt_split=0.1)


def test_cortex():
    run_benchmarks("cortex", n_epochs=1, model=VAEC, tt_split=0.1)


def test_brain_large():
    run_benchmarks("brain_large", n_epochs=1, use_batches=False, tt_split=0.5, unit_test=True)


def test_retina():
    run_benchmarks("retina", n_epochs=1, show_batch_mixing=False, unit_test=True, tt_split=0.1)


def test_ladder():
    gene_dataset = SyntheticDataset()
    tt_split = 0.5
    example_indices = np.random.permutation(len(gene_dataset))
    tt_split = int(tt_split * len(gene_dataset))  # 90%/10% train/test split
    n_epochs = 1
    lr = 1e-3
    data_loader_train = DataLoader(gene_dataset, batch_size=128,
                                   sampler=SubsetRandomSampler(example_indices[:tt_split]),
                                   collate_fn=gene_dataset.collate_fn)
    data_loader_test = DataLoader(gene_dataset, batch_size=128,
                                  sampler=SubsetRandomSampler(example_indices[tt_split:]),
                                  collate_fn=gene_dataset.collate_fn)
    lvaec = LVAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches, n_labels=gene_dataset.n_labels)
    train_semi_supervised(lvaec, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr)

    compute_true_log_likelihood(lvaec, data_loader_train, n_samples=1)

    lvae = LVAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches)
    train(lvae, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr)
    compute_true_log_likelihood(lvae, data_loader_train, n_samples=1)

    dvae = DVAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches)
    train(dvae, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr)
    compute_true_log_likelihood(dvae, data_loader_train, n_samples=1)
