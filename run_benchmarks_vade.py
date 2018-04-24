from torch.utils.data import DataLoader
from scvi.train_vade import train
from scvi.scVADE import VAE
import argparse
import time
from scvi.visualization import show_t_sne
from torch.autograd import Variable
from scvi.dataset import load_datasets
from scvi.clustering import cluster_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250, help="how many times to process the dataset")
    parser.add_argument("--dataset", type=str, default="retina", help="which dataset to process")
    parser.add_argument("--nobatches", action='store_true', help="whether to ignore batches")
    parser.add_argument("--nocuda", action='store_true',
                        help="whether to use cuda (will apply only if cuda is available")

    args = parser.parse_args()
    gene_dataset_train, gene_dataset_test = load_datasets(args.dataset)

    start = time.time()

    data_loader_train = DataLoader(gene_dataset_train, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    data_loader_test = DataLoader(gene_dataset_test, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    vae = VAE(gene_dataset_train.nb_genes, n_clusters=15, batch=True, n_batch=gene_dataset_train.n_batches,
              using_cuda=True)
    if vae.using_cuda:
        vae.cuda()

    train(vae, data_loader_train, data_loader_test, n_epochs=0, learning_rate=1e-3, vade=False)

    # visualizing the latent space at the end of pretraining
    # data_loader_visualize = DataLoader(gene_dataset_test, batch_size=gene_dataset_test.total_size, shuffle=True,
    #                                    num_workers=1, pin_memory=True)
    data_loader_visualize = DataLoader(gene_dataset_test, batch_size=1000, shuffle=True,
                                       num_workers=1, pin_memory=True)
    for i_batch, (sample_batch, _, _, batch_index, c_labels) in enumerate(data_loader_visualize):
        if i_batch == 0:
            sample_batch = Variable(sample_batch)
            if vae.using_cuda:
                sample_batch = sample_batch.cuda(async=True)
                batch_index = batch_index.cuda(async=True)
            px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, ql_m, ql_v = vae(sample_batch, batch_index)
            z = vae.reparameterize(qz_m, qz_v)

    data_loader_clusters = DataLoader(gene_dataset_train, batch_size=gene_dataset_train.total_size, shuffle=True,
                                      num_workers=1, pin_memory=True)
    for i_batch, (sample_batch, _, _, batch_index, c_labels) in enumerate(data_loader_visualize):
        if i_batch == 0:
            sample_batch = Variable(sample_batch)
            if vae.using_cuda:
                sample_batch = sample_batch.cuda(async=True)
                batch_index = batch_index.cuda(async=True)
            px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, ql_m, ql_v = vae(sample_batch, batch_index)
            z = vae.reparameterize(qz_m, qz_v)
            print(cluster_scores(z.data.cpu().numpy(), 15, c_labels.cpu().numpy().flatten()))

    vae.initialize_gmm(data_loader_train)
    train(vae, data_loader_train, data_loader_test, n_epochs=20, learning_rate=1e-3, vade=True)

    # visualizing the latent space of the vade
    for i_batch, (sample_batch, _, _, batch_index, c_labels) in enumerate(data_loader_visualize):
        if i_batch == 0:
            sample_batch = Variable(sample_batch)
            if vae.using_cuda:
                sample_batch = sample_batch.cuda(async=True)
                batch_index = batch_index.cuda(async=True)
            px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, ql_m, ql_v = vae(sample_batch, batch_index)
            z = vae.reparameterize(qz_m, qz_v)
            show_t_sne(z.data.cpu().numpy(), labels=c_labels.cpu().numpy().flatten(), title="After VADE")

    for i_batch, (sample_batch, _, _, batch_index, c_labels) in enumerate(data_loader_visualize):
        if i_batch == 0:
            sample_batch = Variable(sample_batch)
            if vae.using_cuda:
                sample_batch = sample_batch.cuda(async=True)
                batch_index = batch_index.cuda(async=True)
            px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, ql_m, ql_v = vae(sample_batch, batch_index)
            z = vae.reparameterize(qz_m, qz_v)
            print(cluster_scores(z.data.cpu().numpy(), 15, c_labels.cpu().numpy().flatten()))

    end = time.time()
    print("Total runtime for " + str(args.epochs) + " epochs is: " + str((end - start))
          + " seconds for a mean per epoch runtime of " + str((end - start) / args.epochs) + " seconds.")
