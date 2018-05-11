import torch
import torch.nn as nn
from torch.distributions import Normal, Multinomial, kl_divergence as kl

from scvi.metrics.log_likelihood import log_zinb_positive
from scvi.models.modules import Encoder, Classifier, DecoderSCVI, LadderDecoder, LadderEncoder
from scvi.models.utils import enumerate_discrete, one_hot


class LVAEC(nn.Module):
    '''
    "Stacked" variational autoencoder for classification - SVAEC
    (from the stacked generative model M1 + M2)
    '''

    def __init__(self, n_input, n_labels, n_hidden=128, n_latent=10, n_layers=1, dropout_rate=0.1, n_batch=0,
                 y_prior=None, use_cuda=False):
        super(LVAEC, self).__init__()
        self.n_labels = n_labels
        self.n_input = n_input

        self.y_prior = y_prior if y_prior is not None else (1 / self.n_labels) * torch.ones(self.n_labels)
        # Automatically desactivate if useless
        self.n_batch = 0 if n_batch == 1 else n_batch
        self.l_encoder = Encoder(n_input, n_hidden=n_hidden, n_latent=1, n_layers=1,
                                 dropout_rate=dropout_rate)

        n_latent_l = [10,10]
        n_hidden = 128

        self.decoder = DecoderSCVI(n_latent_l[0], n_input, n_hidden=n_hidden, n_layers=n_layers,
                                   dropout_rate=dropout_rate, n_batch=n_batch, n_labels=n_labels)

        self.dispersion = 'gene'
        self.register_buffer('px_r', torch.randn(n_input, ))

        # Classifier takes n_latent as input
        self.classifier = Classifier(n_hidden, n_hidden, n_labels=self.n_labels, n_layers=3)

        self.ladder_encoders = (
            [LadderEncoder(n_input, n_hidden=n_hidden, n_latent=n_latent_l[0], n_layers=n_layers)] +
            [LadderEncoder(n_input=n_hidden, n_hidden=n_hidden, n_latent=n_latent) for n_latent in n_latent_l[1:]])
        self.ladder_decoders = (
            [LadderDecoder(n_latent=n_latent_input, n_output=n_latent_output)
             for (n_latent_input, n_latent_output) in zip(n_latent_l[:0:-1], n_latent_l[-2::-1])]
        )

        for ladder_encoder in  self.ladder_encoders:
            ladder_encoder.cuda()
        for ladder_decoder in  self.ladder_decoders:
            ladder_decoder.cuda()

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.y_prior=self.y_prior.cuda()

    def classify(self, x):
        q = x
        for ladder_encoder in self.ladder_encoders[:-1]:
            _, q = ladder_encoder(q)
        return self.classifier(q)

    def sample_from_posterior_z(self, x, y=None): # which z ? last layer in q (just before classification ?)
        # Here we compute as little as possible to have q(z|x)
        q = x
        for ladder_encoder in self.ladder_encoders[:-1]:
            (_, _, z), q = ladder_encoder(q)
        return z

    def sample_from_posterior_z_out(self, x, y=None): # which z ? last layer in q (just before classification ?)
        # Here we compute as little as possible to have q(z|x)
        q = torch.log(1 + x)
        q_list=[]
        for ladder_encoder in self.ladder_encoders[:-1]:
            (q_m_hat, q_v_hat, z), q = ladder_encoder(q)
            q_list += [(q_m_hat, q_v_hat)]
        if y is None:
            y = self.classifier(q).multinomial(1)
        (_, _, z), q = self.ladder_encoders[-1](q, y)

        for ladder_decoder, (q_m_hat, q_v_hat) in zip(self.ladder_decoders[:-1], reversed(q_list[1:])):
            (_, _, z), _ = ladder_decoder(z, q_m_hat, q_v_hat)
        (_, _, z), _ = self.ladder_decoders[-1](z, q_list[0][0], q_list[0][1], o=y)
        return z

    def sample_from_posterior_l(self, x):
        # Here we compute as little as possible to have q(z|x)
        x = torch.log(1 + x)
        ql_m, ql_v, library = self.l_encoder(x)
        return library

    def get_sample_scale(self, x, y=None, batch_index=None):
        # NOT ! self.sample_from_posterior_z
        z = self.sample_from_posterior_z_out(x, y)
        px = self.decoder.px_decoder(z, batch_index, y)
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_rate(self, x, y=None, batch_index=None):
        # NOT ! self.sample_from_posterior_z
        z = self.sample_from_posterior_z_out(x, y)
        library = self.sample_from_posterior_l(x)
        px = self.decoder.px_decoder(z, batch_index, y)
        return self.decoder.px_scale_decoder(px) * torch.exp(library)

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        is_labelled = False if y is None else True

        xs, ys = (x, y)
        xs_ = torch.log(1 + xs)
        q = xs_

        q_list = []
        for ladder_encoder in self.ladder_encoders[:-1]:
            (q_m_hat, q_v_hat, z), q = ladder_encoder(q)
            q_list += [(q_m_hat, q_v_hat)]

        q_ = q
        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.n_labels)
            xs = xs.repeat(self.n_labels, 1)
            if batch_index is not None:
                batch_index = batch_index.repeat(self.n_labels, 1)
            local_l_var = local_l_var.repeat(self.n_labels, 1)
            local_l_mean = local_l_mean.repeat(self.n_labels, 1)
            q_list = [tuple(q.repeat(self.n_labels, 1) for q in qs) for qs in q_list]
            q = q.repeat(self.n_labels, 1)
        else:
            ys = one_hot(ys, self.n_labels)

        (q_m_hat, q_v_hat, z), q = self.ladder_encoders[-1](q, ys)
        kl_divergence = kl(Normal(q_m_hat, torch.sqrt(q_v_hat)), Normal(torch.zeros_like(q_m_hat), torch.ones_like(q_v_hat))).sum(dim=1)

        reconst_loss = 0

        for ladder_decoder, (q_m_hat, q_v_hat) in zip(self.ladder_decoders, reversed(q_list)):
            (q_m, q_v, z), (p_m, p_v) = ladder_decoder(z, q_m_hat, q_v_hat)
            reconst_loss += (Normal(q_m, torch.sqrt(q_v)).log_prob(z) -  Normal(p_m, torch.sqrt(p_v)).log_prob(z)).sum(dim=1)

        # Sampling
        xs_ = torch.log(1 + xs)

        ql_m, ql_v, library = self.l_encoder(xs_)  # let's keep that ind. of y

        px_scale, px_rate, px_dropout = self.decoder(self.dispersion, z, library, batch_index, y=ys)

        reconst_loss = -log_zinb_positive(xs, px_rate, torch.exp(self.px_r), px_dropout)

        # KL Divergence
        kl_divergence += kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)

        if is_labelled:
            return reconst_loss, kl_divergence

        reconst_loss = reconst_loss.view(self.n_labels, -1)
        kl_divergence = kl_divergence.view(self.n_labels, -1)

        probs = self.classifier(q_)
        reconst_loss = (reconst_loss.t() * probs).sum(dim=1)
        kl_divergence = (kl_divergence.t() * probs).sum(dim=1)
        kl_divergence += kl(Multinomial(probs=probs), Multinomial(probs=self.y_prior))

        return reconst_loss, kl_divergence
