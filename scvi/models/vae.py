# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence as kl

from scvi.metrics.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI

torch.backends.cudnn.benchmark = True


def log_mean_exp(x, axis):
    m = torch.max(x, dim=axis, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m), dim=axis, keepdim=True))


# VAE model
class VAE(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, dispersion="gene",
                 log_variational=True, reconstruction_loss="zinb", n_batch=0, n_labels=0, use_cuda=False):
        super(VAE, self).__init__()
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, ))

        self.z_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)
        self.decoder = DecoderSCVI(n_latent, n_input, n_hidden=n_hidden, n_layers=n_layers,
                                   dropout_rate=dropout_rate, n_batch=n_batch)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def sample_from_posterior_z(self, x, y=None):
        x = torch.log(1 + x)
        # Here we compute as little as possible to have q(z|x)
        qz_m, qz_v, z = self.z_encoder(x)
        return z

    def sample_from_posterior_l(self, x):
        x = torch.log(1 + x)
        # Here we compute as little as possible to have q(z|x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, y=None, batch_index=None):
        x = torch.log(1 + x)
        z = self.sample_from_posterior_z(x)
        px = self.decoder.px_decoder(z, batch_index)
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_rate(self, x, y=None, batch_index=None):
        x = torch.log(1 + x)
        z = self.sample_from_posterior_z(x)
        library = self.sample_from_posterior_l(x)
        px = self.decoder.px_decoder(z, batch_index)
        return self.decoder.px_scale_decoder(px) * torch.exp(library)

    def sample(self, z):
        return self.px_scale_decoder(z)

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):  # same signature as loss
        # Parameters for z latent distribution
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)

        if self.dispersion == "gene-cell":
            px_scale, self.px_r, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)
        elif self.dispersion == "gene":
            px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)

        # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss = -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)
        elif self.reconstruction_loss == 'nb':
            reconst_loss = -log_nb_positive(x, px_rate, torch.exp(self.px_r))

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
        kl_divergence = kl_divergence_z + kl_divergence_l

        return reconst_loss, kl_divergence

    def log_likelihood(self, x, local_l_mean, local_l_var, batch_index=None, n_samples=100):
        x = x.repeat(1, n_samples).view(-1, x.size(1))
        local_l_mean = local_l_mean.repeat(1, n_samples).view(-1, 1)
        local_l_var = local_l_var.repeat(1, n_samples).view(-1, 1)
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)

        ll = 0
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        ll += (- Normal(mean, scale).log_prob(z) +
               Normal(qz_m, torch.sqrt(qz_v)).log_prob(z)).sum(dim=1)

        ll += (- Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library) +
               Normal(ql_m, torch.sqrt(ql_v)).log_prob(library)).sum(dim=1)

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)

        ll += -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)
        ll = ll.view(-1, n_samples)
        return log_mean_exp(ll, -1)
