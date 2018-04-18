import numpy as np
import torch
from torch.utils.data import DataLoader

from scvi.clustering import entropy_batch_mixing, histogram_of_clusters
from scvi.dataset import CortexDataset
from scvi.differential_expression import get_statistics
from scvi.imputation import imputation
from scvi.log_likelihood import compute_log_likelihood
from scvi.scvi import VAE
from scvi.train import train
from scvi.visualization import show_t_sne


def run_benchmarks(gene_dataset_train, gene_dataset_test, n_epochs=1000, learning_rate=1e-3,
                   use_batches=False, use_cuda=True, show_batch_mixing=False):
    # options:
    # - gene_dataset: a GeneExpressionDataset object
    # call each of the 4 benchmarks:
    # - log-likelihood
    # - imputation
    # - batch mixing
    # - cluster scores
    data_loader_train = DataLoader(gene_dataset_train, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    data_loader_test = DataLoader(gene_dataset_test, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    vae = VAE(gene_dataset_train.nb_genes, batch=use_batches, n_batch=gene_dataset_train.n_batches,
              using_cuda=use_cuda)
    if vae.using_cuda:
        vae.cuda()
    train(vae, data_loader_train, data_loader_test, n_epochs=n_epochs, learning_rate=learning_rate)

    # - log-likelihood
    vae.eval()  # Test mode - affecting dropout and batchnorm
    log_likelihood_train = compute_log_likelihood(vae, data_loader_train)
    log_likelihood_test = compute_log_likelihood(vae, data_loader_test)
    print("Log-likelihood Train:", log_likelihood_train)
    print("Log-likelihood Test:", log_likelihood_test)

    # - imputation
    imputation_train = imputation(vae, data_loader_train)
    print("Imputation score on train (MAE) is:", imputation_train)

    # - batch mixing
    if gene_dataset_train.n_batches >= 2:
        latent = []
        batch_indices = []
        for sample_batch, local_l_mean, local_l_var, batch_index, _ in data_loader_train:
            if vae.using_cuda:
                sample_batch = sample_batch.cuda(async=True)
            latent += [vae.sample_from_posterior(sample_batch)]  # Just run a forward pass on all the data
            batch_indices += [batch_index]
        latent = torch.cat(latent)
        batch_indices = torch.cat(batch_indices)

    if gene_dataset_train.n_batches == 2:
        print("Entropy batch mixing :", entropy_batch_mixing(latent.data.cpu().numpy(), batch_indices.numpy()))
    if show_batch_mixing:
        show_t_sne(latent.data.cpu().numpy(), np.array([batch[0] for batch in batch_indices.numpy()]),
                   "Batch mixing t_SNE plot")

    # - differential expression
    #
    if type(gene_dataset_train) == CortexDataset:
        get_statistics(vae, data_loader_train, M_sampling=1, M_permutation=1)  # 200 - 100000
        histogram_of_clusters(vae, data_loader_train, filename="clustering_10.png")
