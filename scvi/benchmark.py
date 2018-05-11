import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from scvi.metrics.classification import compute_accuracy_svc, compute_accuracy_rf,\
    compute_accuracy_md, compute_accuracy_classes
from sklearn.decomposition import PCA
from scvi.dataset import CortexDataset, load_datasets
from scvi.metrics.adapt_encoder import adapt_encoder
from scvi.metrics.clustering import entropy_batch_mixing, get_latent
from scvi.metrics.differential_expression import get_statistics
from scvi.metrics.imputation import imputation
from scvi.metrics.visualization import show_t_sne
from scvi.models import VAE, SVAEC
from scvi.models.modules import Classifier
from scvi.train import train, train_classifier, train_semi_supervised


def run_benchmarks(dataset_name, model=VAE, n_epochs=1000, lr=1e-3, use_batches=False, use_cuda=True,
                   show_batch_mixing=True, benchmark=False, tt_split=0.9, unit_test=False):
    # options:
    # - gene_dataset: a GeneExpressionDataset object
    # call each of the 4 benchmarks:
    # - log-likelihood
    # - imputation
    # - batch mixing
    # - cluster scores
    gene_dataset = load_datasets(dataset_name, unit_test=unit_test)
    example_indices = np.random.permutation(len(gene_dataset))
    tt_split = int(tt_split * len(gene_dataset))  # 90%/10% train/test split

    data_loader_train = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,
                                   sampler=SubsetRandomSampler(example_indices[:tt_split]),
                                   collate_fn=gene_dataset.collate_fn)
    data_loader_test = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,
                                  sampler=SubsetRandomSampler(example_indices[tt_split:]),
                                  collate_fn=gene_dataset.collate_fn)
    vae = model(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches, n_labels=gene_dataset.n_labels,
                use_cuda=use_cuda)
    stats = train(vae, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr, benchmark=benchmark)

    if isinstance(vae, VAE):
        best_ll = adapt_encoder(vae, data_loader_test, n_path=1, n_epochs=1, record_freq=1)
        print("Best ll was :", best_ll)

    # - log-likelihood
    print("Log-likelihood Train:", stats.history["LL_train"][stats.best_index])
    print("Log-likelihood Test:", stats.history["LL_test"][stats.best_index])

    # - imputation
    imputation_test = imputation(vae, data_loader_test)
    print("Imputation score on test (MAE) is:", imputation_test.item())

    # - batch mixing
    if gene_dataset.n_batches == 2:
        latent, batch_indices, labels = get_latent(vae, data_loader_train)
        print("Entropy batch mixing :", entropy_batch_mixing(latent.cpu().numpy(), batch_indices.cpu().numpy()))
        if show_batch_mixing:
            show_t_sne(latent.cpu().numpy(), np.array([batch[0] for batch in batch_indices.cpu().numpy()]))

    # - differential expression
    if type(gene_dataset) == CortexDataset:
        get_statistics(vae, data_loader_train, M_sampling=1, M_permutation=1)  # 200 - 100000


# Pipeline to compare different semi supervised models
def run_benchmarks_classification(dataset_name, n_latent=10, n_epochs=10, n_epochs_classifier=10, lr=1e-2,
                                  use_batches=False, use_cuda=True, tt_split=0.1):
    gene_dataset = load_datasets(dataset_name)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    alpha = 100  # in Kingma, 0.1 * len(gene_dataset), but pb when : len(gene_dataset) >> 1

    # Create the dataset
    # example_indices = np.random.permutation(len(gene_dataset))
    # probabilities = [1 / gene_dataset.n_labels for i in range(gene_dataset.n_labels)]
    probabilities = [0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05]
    example_indices = gene_dataset.get_indices(probabilities, scale=tt_split)
    # example_indices = np.random.permutation(len(gene_dataset))

    tt_split = int(tt_split * len(gene_dataset))
    train_indices = example_indices[:tt_split]
    test_indices = example_indices[tt_split:]
    data_loader_train = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,
                                   sampler=SubsetRandomSampler(train_indices),
                                   collate_fn=gene_dataset.collate_fn)
    data_loader_test = DataLoader(gene_dataset, batch_size=128, pin_memory=use_cuda,
                                  sampler=SubsetRandomSampler(test_indices),
                                  collate_fn=gene_dataset.collate_fn)

    # We start by trying classic ML techniques to use them as benchmarks

    # ========= SVM and Decison Tree baseline ==========

    print("Baseline with SVM, Decision tree and majority decision ")
    data_train, labels_train = gene_dataset.data_for_classification(train_indices)
    data_test, labels_test = gene_dataset.data_for_classification(test_indices)

    # PCA then majority decision
    X = np.concatenate((data_train, data_test))
    pca = PCA(n_components=10)
    pca.fit_transform(X)
    latent_train = pca.transform(data_train)
    latent_test = pca.transform(data_test)

    # Majority decision on the VAE's latent space using Kmeans for clustering
    accuracy_train, accuracy_test = compute_accuracy_md(latent_train,
                                                        latent_test,
                                                        labels_train,
                                                        labels_test,
                                                        n_labels=gene_dataset.n_labels)
    print(accuracy_test)
    axes[0].plot(np.repeat(accuracy_train, n_epochs), '--', label='Clustering baseline')
    axes[1].plot(np.repeat(accuracy_train, n_epochs), '--')

    accuracy_train_svc, accuracy_test_svc = compute_accuracy_svc(data_train, data_test, labels_train, labels_test,
                                                                 unit_test=True)
    print(accuracy_test_svc)
    axes[0].plot(np.repeat(accuracy_train_svc, n_epochs), label='SVC')
    axes[1].plot(np.repeat(accuracy_test_svc, n_epochs))

    accuracy_train_dt, accuracy_test_dt = compute_accuracy_rf(data_train, data_test, labels_train, labels_test,
                                                              unit_test=True)
    print(accuracy_test_dt)
    axes[0].plot(np.repeat(accuracy_train_dt, n_epochs), label='RF')
    axes[1].plot(np.repeat(accuracy_test_dt, n_epochs))

    # Now we try out the different models and compare their accuracy

    # ========== The M1 model ===========
    print("Trying M1 model")
    vae = VAE(gene_dataset.nb_genes, n_latent=n_latent,
              n_batch=gene_dataset.n_batches * use_batches, use_cuda=use_cuda,
              n_labels=gene_dataset.n_labels)
    train(vae, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr)
    # Then we train a classifier on the latent space
    cls = Classifier(n_input=n_latent, n_labels=gene_dataset.n_labels, n_layers=3, use_cuda=use_cuda)
    for param in vae.z_encoder.parameters():
        param.requires_grad = False
    cls_stats = train_classifier(vae, cls, data_loader_train, data_loader_test, n_epochs=n_epochs_classifier, lr=lr)
    print(compute_accuracy_classes(vae, data_loader_train, cls))

    axes[0].plot(cls_stats.history["Accuracy_train"], label='VAE + classifier')
    axes[1].plot(cls_stats.history["Accuracy_test"])

    axes[0].plot(cls_stats.history["Weighted_accuracy_train"], label='VAE + classifier (weighted')
    axes[1].plot(cls_stats.history["Weighted_accuracy_test"])

    axes[0].plot(cls_stats.history["Worst_accuracy_train"], label='VAE + classifier (Worst)')
    axes[1].plot(cls_stats.history["Worst_accuracy_test"])

    # ========== Majority decision ========

    # Majority decision on the VAE's latent space using Kmeans for clustering
    # print("Majority decision on latent space")
    # latent_train, batch_indices_train, labels_train = get_latent(vae, data_loader_train)
    # latent_test, batch_indices_test, labels_test = get_latent(vae, data_loader_test)
    # accuracy_train, accuracy_test = compute_accuracy_md(latent_train.cpu().numpy(),
    #                                                     latent_test.cpu().numpy(),
    #                                                     labels_train.cpu().numpy(),
    #                                                     labels_test.cpu().numpy(),
    #                                                     n_labels=gene_dataset.n_labels)
    #
    # axes[0].plot(np.repeat(accuracy_train, n_epochs), '--', label='Clustering baseline')
    # axes[1].plot(np.repeat(accuracy_train, n_epochs), '--')

    # ========== The VAEC model ===========

    # print("Trying VAEC model")
    prior = torch.FloatTensor([(gene_dataset.labels == i).mean() for i in range(gene_dataset.n_labels)])
    #
    # vaec = VAEC(gene_dataset.nb_genes, n_labels=gene_dataset.n_labels, y_prior=prior,
    #             n_latent=n_latent, use_cuda=use_cuda)
    #
    # stats = train_semi_supervised(vaec, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr,
    #                               classification_ratio=alpha)
    #
    # axes[0].plot(stats.history["Accuracy_train"], label='VAEC')
    # axes[1].plot(stats.history["Accuracy_test"])

    # ========== The M1+M2 model trained jointly ===========
    print("Trying out M1+M2 optimized jointly")
    svaec = SVAEC(gene_dataset.nb_genes, n_labels=gene_dataset.n_labels, y_prior=prior, n_latent=n_latent,
                  use_cuda=use_cuda)

    stats = train_semi_supervised(svaec, data_loader_train, data_loader_test, n_epochs=n_epochs, lr=lr,
                                  classification_ratio=alpha)

    axes[0].plot(stats.history["Accuracy_train"], label='M1+M2 (train all)')
    axes[1].plot(stats.history["Accuracy_test"])

    # ========== Classifier trained on the latent space of M1+M2 ===========
    # print("Trying to classify on M1+M2's z1 latent space")
    # cls = Classifier(n_input=n_latent, n_labels=gene_dataset.n_labels, n_layers=3, use_cuda=use_cuda)
    #
    # stats = train_classifier(svaec, cls, data_loader_train, data_loader_test, n_epochs=n_epochs_classifier, lr=lr)
    #
    # axes[0].plot(stats.history["Accuracy_train"], label='M1+M2+classifier')
    # axes[1].plot(stats.history["Accuracy_test"])

    # Now plot the results
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('n_epochs')
    axes[0].set_title('acc. train')
    axes[0].legend()
    axes[1].set_xlabel('n_epochs')
    axes[1].set_title('acc. test')

    plt.tight_layout()
    plt.savefig("result_classification.png")
