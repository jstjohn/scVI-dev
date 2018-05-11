import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence as kl

from scvi.metrics.log_likelihood import log_zinb_positive
from scvi.models.modules import Encoder, DecoderSCVI, Decoder, LadderEncoder, LadderDecoder
from scvi.models.modules import ListModule
from scvi.models.utils import log_mean_exp


class DVAE(nn.Module):
    '''
    Deep VAE : multiple layers of stochastic variable
    '''

    def __init__(self, n_input, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, n_batch=0, use_cuda=False):
        super(DVAE, self).__init__()
        self.n_input = n_input

        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)

        n_latent_l = [15, 10]
        n_hidden = 128

        self.decoder = DecoderSCVI(n_latent_l[0], n_input, n_hidden=n_hidden, n_layers=1,
                                   dropout_rate=dropout_rate, n_batch=n_batch)

        self.dispersion = 'gene'
        self.px_r = torch.nn.Parameter(torch.randn(n_input, ))

        self.ladder_encoders = ListModule(
            *([Encoder(n_input, n_hidden=n_hidden, n_latent=n_latent_l[0], n_layers=n_layers)] +
              [Encoder(n_input=n_latent_in, n_hidden=n_hidden, n_latent=n_latent_out) for n_latent_in, n_latent_out
               in zip(n_latent_l[:-1], n_latent_l[1:])]))
        self.ladder_decoders = ListModule(
            *([Decoder(n_latent=n_latent_input, n_output=n_latent_output)
               for (n_latent_input, n_latent_output) in zip(n_latent_l[:0:-1], n_latent_l[-2::-1])])
        )

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        x_ = torch.log(1 + x)
        z = x_
        z_list = []
        kl_divergence = 0

        ql_m, ql_v, library = self.l_encoder(x_)  # let's keep that ind. of y
        kl_divergence += kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        for ladder_encoder in self.ladder_encoders:
            (q_m_hat, q_v_hat, z) = ladder_encoder(z)
            kl_divergence += (Normal(q_m_hat, torch.sqrt(q_v_hat)).log_prob(z)).sum(dim=1)
            z_list += [z]

        kl_divergence -= (Normal(torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat)).log_prob(z)).sum(dim=-1)

        for ladder_decoder, z_down, z_up in zip(self.ladder_decoders, reversed(z_list[:-1]), reversed(z_list[1:])):
            p_m, p_v = ladder_decoder(z_up)
            kl_divergence += -(Normal(p_m, torch.sqrt(p_v)).log_prob(z_down)).sum(dim=1)

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z_down, library, batch_index)

        reconst_loss = -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)

        return reconst_loss, kl_divergence

    def log_likelihood(self, x, local_l_mean, local_l_var, batch_index=None, labels=None, n_samples=1):
        x = x.repeat(1, n_samples).view(-1, x.size(1))

        local_l_mean = local_l_mean.repeat(1, n_samples).view(-1, 1)
        local_l_var = local_l_var.repeat(1, n_samples).view(-1, 1)
        x_ = torch.log(1 + x)
        z = x_
        ll = 0
        ql_m, ql_v, library = self.l_encoder(x_)

        ll += (- Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library) +
               Normal(ql_m, torch.sqrt(ql_v)).log_prob(library)).sum(dim=1)

        z_list = []

        for ladder_encoder in self.ladder_encoders:
            (q_m_hat, q_v_hat, z) = ladder_encoder(z)
            ll += (Normal(q_m_hat, torch.sqrt(q_v_hat)).log_prob(z)).sum(dim=1)
            z_list += [z]

        ll -= (Normal(torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat)).log_prob(z)).sum(dim=1)

        for ladder_decoder, z_down, z_up in zip(self.ladder_decoders, reversed(z_list[:-1]), reversed(z_list[1:])):
            p_m, p_v = ladder_decoder(z_up)
            ll += -(Normal(p_m, torch.sqrt(p_v)).log_prob(z_down)).sum(dim=1)

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z_down, library, batch_index)

        ll += -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)
        ll = ll.view(-1, n_samples)
        return log_mean_exp(ll, -1)


class LVAE(nn.Module):
    '''
    Ladder VAE : multiple layers of stochastic variable
    Instead of having q(z1|z2), we have q(z1|x)
    '''

    def __init__(self, n_input, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, n_batch=0, use_cuda=False):
        super(LVAE, self).__init__()
        self.n_input = n_input

        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)

        n_latent_l = [64, 32, 16]
        n_hidden = 128

        self.decoder = DecoderSCVI(n_latent_l[0], n_input, n_hidden=n_hidden, n_layers=1,
                                   dropout_rate=dropout_rate, n_batch=n_batch)

        self.dispersion = 'gene'
        self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        self.ladder_encoders = ListModule(
            *([LadderEncoder(n_input, n_hidden=n_hidden, n_latent=n_latent_l[0], n_layers=n_layers)] +
              [LadderEncoder(n_input=n_hidden, n_hidden=n_hidden, n_latent=n_latent) for n_latent in n_latent_l[1:]]))
        self.ladder_decoders = ListModule(
            *([LadderDecoder(n_latent=n_latent_input, n_output=n_latent_output)
               for (n_latent_input, n_latent_output) in zip(n_latent_l[:0:-1], n_latent_l[-2::-1])])
        )

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        x_ = torch.log(1 + x)
        q = x_
        q_list = []
        kl_divergence = 0
        reconst_loss = 0

        # latent variable l
        ql_m, ql_v, library = self.l_encoder(x_)  # let's keep that ind. of y
        kl_divergence += kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        # latent variable z
        for ladder_encoder in self.ladder_encoders:
            (q_m_hat, q_v_hat, z), q = ladder_encoder(q)
            q_list += [(q_m_hat, q_v_hat)]

        mean, var = torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat)
        kl_divergence += kl(Normal(q_m_hat, torch.sqrt(q_v_hat)), Normal(mean, torch.sqrt(var))).sum(dim=1)

        for ladder_decoder, (q_m_hat, q_v_hat) in zip(self.ladder_decoders, reversed(q_list[:-1])):
            (q_m, q_v, z), (p_m, p_v) = ladder_decoder(z, q_m_hat, q_v_hat)
            kl_divergence += (Normal(q_m, torch.sqrt(q_v)).log_prob(z)
                              - Normal(p_m, torch.sqrt(p_v)).log_prob(z)).sum(dim=1)

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)
        reconst_loss += -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)

        return reconst_loss, kl_divergence

    def log_likelihood(self, x, local_l_mean, local_l_var, batch_index=None, labels=None, n_samples=100):
        x = x.repeat(1, n_samples).view(-1, x.size(1))
        local_l_mean = local_l_mean.repeat(1, n_samples).view(-1, 1)
        local_l_var = local_l_var.repeat(1, n_samples).view(-1, 1)

        x_ = torch.log(1 + x)

        ll = 0

        ql_m, ql_v, library = self.l_encoder(x_)
        ll += (- Normal(local_l_mean, torch.sqrt(local_l_var)).log_prob(library) +
               Normal(ql_m, torch.sqrt(ql_v)).log_prob(library)).sum(dim=1)

        q = x_
        q_list = []
        for ladder_encoder in self.ladder_encoders:
            (q_m_hat, q_v_hat, z), q = ladder_encoder(q)
            q_list += [(q_m_hat, q_v_hat)]

        ll += (Normal(q_m_hat, torch.sqrt(q_v_hat)).log_prob(z) -
               Normal(torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat)).log_prob(z)).sum(dim=1)

        for ladder_decoder, (q_m_hat, q_v_hat) in zip(self.ladder_decoders, reversed(q_list[:-1])):
            (q_m, q_v, z), (p_m, p_v) = ladder_decoder(z, q_m_hat, q_v_hat)
            ll += (Normal(q_m, torch.sqrt(q_v)).log_prob(z) - Normal(p_m, torch.sqrt(p_v)).log_prob(z)).sum(dim=1)

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index)
        ll += -log_zinb_positive(x, px_rate, torch.exp(self.px_r), px_dropout)

        ll = ll.view(-1, n_samples)
        return log_mean_exp(ll, -1)
