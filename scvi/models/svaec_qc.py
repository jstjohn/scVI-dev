# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Multinomial, kl_divergence as kl

from scvi.metrics.log_likelihood import log_zinb_positive, log_nb_positive, log_bernoulli_with_logits, hsic_objective
from scvi.models.modules import Encoder, DecoderSCVIQC, Classifier, Decoder
from scvi.models.utils import one_hot, enumerate_discrete

torch.backends.cudnn.benchmark = True


# VAE model
class SVAECQC(nn.Module):
    def __init__(self, n_input, n_input_qc, n_hidden=128, n_latent=10, n_latent_qc=3, n_layers=1, dropout_rate=0.1,
                 dispersion="gene", log_variational=True, reconstruction_loss="zinb", n_batch=0, n_labels=0,
                 n_s_decoder=25, use_cuda=False, y_prior=None):
        super(SVAECQC, self).__init__()
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        self.n_labels = n_labels

        self.y_prior = y_prior if y_prior is not None else (1 / self.n_labels) * torch.ones(self.n_labels)
        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        self.z_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.u_encoder = Encoder(n_input+n_input_qc, n_hidden=n_hidden, n_latent=n_latent_qc, n_layers=n_layers,
                                 dropout_rate=dropout_rate)
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)
        self.decoder = DecoderSCVIQC(n_latent, n_latent_qc, n_input, n_input_qc, n_hidden=n_hidden, n_layers=n_layers,
                                     dropout_rate=dropout_rate, n_batch=n_batch, n_s_decoder=n_s_decoder)

        self.classifier = Classifier(n_latent, n_hidden, self.n_labels, n_layers, dropout_rate)
        self.encoder_z2_z1 = Encoder(n_input=n_latent, n_cat=self.n_labels, n_latent=n_latent, n_layers=n_layers)
        self.decoder_z1_z2 = Decoder(n_latent, n_latent, n_cat=self.n_labels, n_layers=n_layers)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.y_prior = self.y_prior.cuda()

    def classify(self, x):
        x_ = torch.log(1 + x)
        qz_m, _, z = self.z_encoder(x_)

        if self.training:
            return self.classifier(z)
        else:
            return self.classifier(qz_m)

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
        z = self.sample_from_posterior_z(x, y)
        u = self.sample_from_posterior_u(x, qc, y)
        px = self.decoder.px_decoder(torch.cat((z, u), 1), batch_index, y)
        px_scale = self.decoder.px_scale_decoder(torch.cat((px, u), 1))
        return px_scale

    def get_sample_rate(self, x, qc, y=None, batch_index=None, eps=1e-8):
        x_ = torch.log(1 + x)
        library = self.sample_from_posterior_l(x_)
        px_scale = self.get_sample_scale(x=x, qc=qc, y=y, batch_index=batch_index, eps=eps)
        return px_scale * torch.exp(library)

    def forward(self, x, local_l_mean, local_l_var, qc, batch_index=None, y=None, eps=1e-8):  # same signature as loss
        is_labelled = False if y is None else True
        # Parameters for z latent distribution
        x_ = x
        y_ = y
        qc_ = torch.log(eps + qc) - torch.log(1 - qc + eps)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz1_m, qz1_v, z1 = self.z_encoder(x_)
        qu_m, qu_v, u = self.u_encoder(torch.cat((x_, qc_), 1))
        ql_m, ql_v, library = self.l_encoder(x_)

        # Enumerate choices of label
        if not is_labelled:
            y_ = enumerate_discrete(x_, self.n_labels)
            x = x.repeat(self.n_labels, 1)
            if batch_index is not None:
                batch_index = batch_index.repeat(self.n_labels, 1)
            local_l_var = local_l_var.repeat(self.n_labels, 1)
            local_l_mean = local_l_mean.repeat(self.n_labels, 1)
            qz1_m = qz1_m.repeat(self.n_labels, 1)
            qz1_v = qz1_v.repeat(self.n_labels, 1)
            qu_m = qu_m.repeat(self.n_labels, 1)
            qu_v = qu_v.repeat(self.n_labels, 1)
            ql_m = ql_m.repeat(self.n_labels, 1)
            ql_v = ql_v.repeat(self.n_labels, 1)
            z1 = z1.repeat(self.n_labels, 1)
            u = u.repeat(self.n_labels, 1)
            library = library.repeat(self.n_labels, 1)
            qc = qc.repeat(self.n_labels, 1)
        else:
            y_ = one_hot(y_, self.n_labels)

        if self.dispersion == "gene-cell":
            px_scale, self.px_r, px_rate, px_dropout, ps_logit = self.decoder(self.dispersion, z1, u, library,
                                                                              batch_index, y=y_)
        else:  # self.dispersion == "gene", "gene-batch",  "gene-label"
            px_scale, px_rate, px_dropout, ps_logit = self.decoder(self.dispersion, z1, u, library, batch_index, y=y_)

        if self.dispersion == "gene-label":
            px_r = F.linear(y_, self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        else:
            px_r = self.px_r

        # Reconstruction Loss
        if self.reconstruction_loss == 'zinb':
            reconst_loss = -log_zinb_positive(x, px_rate, torch.exp(px_r), px_dropout)
        elif self.reconstruction_loss == 'nb':
            reconst_loss = -log_nb_positive(x, px_rate, torch.exp(px_r))

        qz2_m, qz2_v, z2 = self.encoder_z2_z1(z1, y_)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, y_)

        reconst_loss -= log_bernoulli_with_logits(qc, ps_logit)

        # KL Divergence
        mean_z1 = torch.zeros_like(qz1_m)
        scale_z1 = torch.ones_like(qz1_v)
        mean_z2 = torch.zeros_like(qz2_m)
        scale_z2 = torch.ones_like(qz2_v)
        mean_u = torch.zeros_like(qu_m)
        scale_u = torch.ones_like(qu_v)

        kl_divergence_z2 = kl(Normal(qz2_m, torch.sqrt(qz2_v)), Normal(mean_z2, scale_z2)).sum(dim=1)
        loss_z1 = (- Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1) +
                   Normal(qz1_m, torch.sqrt(qz1_v)).log_prob(z1)).sum(dim=1)

        kl_divergence_z1 = kl(Normal(qz1_m, torch.sqrt(qz1_v)), Normal(mean_z1, scale_z1)).sum(dim=1)
        kl_divergence_u = kl(Normal(qu_m, torch.sqrt(qu_v)), Normal(mean_u, scale_u)).sum(dim=1)
        kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
        hsic_loss = hsic_objective(z1, u)
        kl_divergence = kl_divergence_z1 + kl_divergence_l + kl_divergence_u + hsic_loss + loss_z1 + kl_divergence_z2

        if is_labelled:
            return reconst_loss, kl_divergence

        probs = self.classifier(z1)
        reconst_loss = (reconst_loss[:, None] * probs).sum(dim=1)
        kl_divergence = (kl_divergence[:, None] * probs).sum(dim=1)

        kl_divergence += kl(Multinomial(probs=probs), Multinomial(probs=self.y_prior))

        return reconst_loss, kl_divergence
