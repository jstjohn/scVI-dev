# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.metrics.log_likelihood import (
    log_zinb_positive,
    log_nb_positive,
    log_bernoulli_with_logits,
    hsic_objective,
    calculate_gamma_given_latent_dim,
)
from scvi.models.modules import Encoder, DecoderSCVIQC
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True


# VAE model
class VAEQC(nn.Module):
    def __init__(self, n_input, n_input_qc, n_hidden=128, n_latent=10, n_latent_qc=3, n_layers=1, dropout_rate=0.1,
                 dispersion="gene", log_variational=True, reconstruction_loss="zinb", n_batch=0, n_labels=0,
                 n_s_decoder=25, use_cuda=False):
        super(VAEQC, self).__init__()
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.n_labels = n_labels

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        self.gamma_z = torch.tensor(calculate_gamma_given_latent_dim(n_latent), requires_grad=False)
        self.gamma_u = torch.tensor(calculate_gamma_given_latent_dim(n_latent_qc), requires_grad=False)

        self.z_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.u_encoder = Encoder(n_input+n_input_qc, n_hidden=n_hidden, n_latent=n_latent_qc, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)
        self.decoder = DecoderSCVIQC(n_latent, n_latent_qc, n_input, n_input_qc, n_hidden=n_hidden, n_layers=n_layers,
                                     dropout_rate=dropout_rate, n_batch=n_batch, n_s_decoder=n_s_decoder)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.gamma_u = self.gamma_u.cuda()
            self.gamma_z = self.gamma_z.cuda()

    def sample_from_posterior_u(self, x, qc, y=None, eps=1e-8):
        x = torch.log(1 + x)
        qc = torch.log(eps + qc) - torch.log(1 - qc + eps)
        # Here we compute as little as possible to have q(z|x)
        qu_m, qu_v, u = self.u_encoder(torch.cat((x, qc), 1))
        return u

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

    def get_sample_scale(self, x, qc, y=None, batch_index=None, eps=1e-8):
        x = torch.log(1 + x)
        qc = torch.log(eps + qc) - torch.log(1 - qc + eps)
        z = self.sample_from_posterior_z(x)
        u = self.sample_from_posterior_u(x, qc)
        px = self.decoder.px_decoder(torch.cat((z, u), 1), batch_index)
        px_scale = self.decoder.px_scale_decoder(torch.cat((px, u), 1))
        return px_scale

    def get_sample_rate(self, x, qc, y=None, batch_index=None, eps=1e-8):
        x_ = torch.log(1 + x)
        library = self.sample_from_posterior_l(x_)
        px_scale = self.get_sample_scale(x=x, qc=qc, y=y, batch_index=batch_index, eps=eps)
        return px_scale * torch.exp(library)

    def forward(self, x, local_l_mean, local_l_var, qc, batch_index=None, y=None, eps=1e-8):  # same signature as loss
        # Parameters for z latent distribution
        x_ = x
        qc_ = torch.log(eps + qc) - torch.log(1 - qc + eps)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_)
        qu_m, qu_v, u = self.u_encoder(torch.cat((x_, qc_), 1))
        ql_m, ql_v, library = self.l_encoder(x_)

        if self.dispersion == "gene-cell":
            px_scale, self.px_r, px_rate, px_dropout, ps_logit = self.decoder(self.dispersion, z, u, library,
                                                                              batch_index)
        else:  # self.dispersion == "gene", "gene-batch",  "gene-label"
            px_scale, px_rate, px_dropout, ps_logit = self.decoder(self.dispersion, z, u, library, batch_index)

        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        else:
            px_r = self.px_r

        # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss = -log_zinb_positive(x, px_rate, torch.exp(px_r), px_dropout)
        elif self.reconstruction_loss == 'nb':
            reconst_loss = -log_nb_positive(x, px_rate, torch.exp(px_r))

        reconst_loss -= log_bernoulli_with_logits(qc, ps_logit)

        # KL Divergence
        mean_z = torch.zeros_like(qz_m)
        scale_z = torch.ones_like(qz_v)
        mean_u = torch.zeros_like(qu_m)
        scale_u = torch.ones_like(qu_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean_z, scale_z)).sum(dim=1)
        kl_divergence_u = kl(Normal(qu_m, torch.sqrt(qu_v)), Normal(mean_u, scale_u)).sum(dim=1)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
        hsic_loss = hsic_objective(z, u, gamma_z=self.gamma_z, gamma_u=self.gamma_u)
        kl_divergence = kl_divergence_z + kl_divergence_l + kl_divergence_u + hsic_loss

        return reconst_loss, kl_divergence
