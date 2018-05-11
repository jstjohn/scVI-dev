import torch
import torch.nn as nn
from torch.distributions import Normal, Multinomial, kl_divergence as kl

from scvi.metrics.log_likelihood import log_zinb_positive
from scvi.models.modules import Encoder, Classifier, DecoderSCVI, LadderDecoder, LadderEncoder, ListModule
from scvi.models.utils import broadcast_labels


class LVAEC(nn.Module):
    '''
    Ladder VAE for classification: multiple layers of stochastic variable
    Instead of having q(z1|z2), we have q(z1|x)
    '''

    def __init__(self, n_input, n_labels, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, n_batch=0,
                 use_cuda=False, y_prior=None):
        super(LVAEC, self).__init__()
        self.n_input = n_input
        self.n_labels = n_labels
        self.y_prior = y_prior if y_prior is not None else (1 / self.n_labels) * torch.ones(self.n_labels)
        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)

        n_latent_l = [64, 32, 16]
        n_hidden = 128

        self.decoder = DecoderSCVI(n_latent_l[0], n_input, n_hidden=n_hidden, n_layers=1,
                                   dropout_rate=dropout_rate, n_batch=n_batch, n_labels=n_labels)

        self.classifier = Classifier(n_hidden, n_hidden, n_labels=self.n_labels, n_layers=3)

        self.dispersion = 'gene'
        self.px_r = torch.nn.Parameter(torch.randn(n_input, ))
        self.ladder_encoders = ListModule(
            *([LadderEncoder(n_input, n_hidden=n_hidden, n_latent=n_latent_l[0], n_layers=n_layers, n_cat=n_labels)] +
              [LadderEncoder(n_input=n_hidden, n_hidden=n_hidden, n_latent=n_latent, n_cat=n_labels) for n_latent in
               n_latent_l[1:]]))
        self.ladder_decoders = ListModule(
            *([LadderDecoder(n_latent=n_latent_input, n_output=n_latent_output, n_cat=n_labels)
               for (n_latent_input, n_latent_output) in zip(n_latent_l[:0:-1], n_latent_l[-2::-1])])
        )

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.y_prior = self.y_prior.cuda()

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        is_labelled = False if y is None else True

        (xs, ys) = (x, y)
        x_ = torch.log(1 + xs)

        kl_divergence = 0
        reconst_loss = 0

        # KL Divergence
        ql_m, ql_v, library = self.l_encoder(x_)  # let's keep that ind. of y
        kl_divergence += kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        ys, xs, batch_index, library, kl_divergence = \
            broadcast_labels(ys, xs, batch_index, library, kl_divergence, n_broadcast=self.n_labels)

        q = torch.log(1 + xs)
        q_list = []
        # latent variable z
        for i, ladder_encoder in enumerate(self.ladder_encoders):
            (q_m_hat, q_v_hat, z), q = ladder_encoder(q, o=ys)
            q_list += [(q_m_hat, q_v_hat)]
            if i == 0:
                q_pred = q  # only the first q used for the prediction of y

        mean, var = torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat)
        kl_divergence += kl(Normal(q_m_hat, torch.sqrt(q_v_hat)), Normal(mean, torch.sqrt(var))).sum(dim=1)

        for ladder_decoder, (q_m_hat, q_v_hat) in zip(self.ladder_decoders, reversed(q_list[:-1])):
            (q_m, q_v, z), (p_m, p_v) = ladder_decoder(z, q_m_hat, q_v_hat, o=ys)
            kl_divergence += (Normal(q_m, torch.sqrt(q_v)).log_prob(z)
                              - Normal(p_m, torch.sqrt(p_v)).log_prob(z)).sum(dim=1)

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index, y=ys)
        reconst_loss += -log_zinb_positive(xs, px_rate, torch.exp(self.px_r), px_dropout)

        if is_labelled:
            return reconst_loss, kl_divergence

        reconst_loss = reconst_loss.view(self.n_labels, -1)
        kl_divergence = kl_divergence.view(self.n_labels, -1)

        probs = self.classifier(q_pred[:x.size(0)])
        reconst_loss = (reconst_loss.t() * probs).sum(dim=1)
        kl_divergence = (kl_divergence.t() * probs).sum(dim=1)
        kl_divergence += kl(Multinomial(probs=probs), Multinomial(probs=self.y_prior))

        return reconst_loss, kl_divergence
