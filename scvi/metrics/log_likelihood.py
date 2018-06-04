"""File for computing log likelihood of the data"""

import torch
import torch.nn.functional as F

from scvi.utils import to_cuda, no_grad, eval_modules


@no_grad()
@eval_modules()
def compute_log_likelihood(vae, data_loader):
    # Iterate once over the data_loader and computes the total log_likelihood
    log_lkl = 0
    for i_batch, tensors in enumerate(data_loader):
        if vae.use_cuda:
            tensors = to_cuda(tensors)
        sample_batch, sample_qc, local_l_mean, local_l_var, batch_index, labels = tensors
        sample_batch = sample_batch.type(torch.float32)
        reconst_loss, kl_divergence = vae(sample_batch, local_l_mean, local_l_var, qc=sample_qc,
                                          batch_index=batch_index, y=labels)
        log_lkl += torch.sum(reconst_loss).item()
    n_samples = (len(data_loader.dataset)
                 if not (hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'indices')) else
                 len(data_loader.sampler.indices))
    return log_lkl / n_samples


def sigmoid_cross_entropy_with_logits(x, z):
    return torch.clamp(x, min=0.0) - x * z + torch.log(1 + torch.exp(-torch.abs(x)))


def hsic_objective(z, s, use_cuda=True):
    # use a gaussian RBF for every variable
    def K(x1, x2, gamma=1.):
        dist_table = x1[None, :] - x2[:, None]
        return torch.t(torch.exp(torch.clamp(-gamma * torch.sum(dist_table ** 2, 2), max=20)))

    n_latent_z = torch.tensor(z.size(1), dtype=torch.float)
    n_latent_s = torch.tensor(s.size(1), dtype=torch.float)

    if use_cuda:
        n_latent_s = n_latent_s.cuda()
        n_latent_z = n_latent_z.cuda()

    gz = 2 * torch.exp(torch.lgamma(0.5 * (n_latent_z + 1)) - torch.lgamma(0.5 * n_latent_z))
    gs = 2 * torch.exp(torch.lgamma(0.5 * (n_latent_s + 1)) - torch.lgamma(0.5 * n_latent_s))

    zz = K(z, z, gamma=1. / (2. * gz))
    ss = K(s, s, gamma=1. / (2. * gs))

    hsic = 0
    hsic += torch.mean(zz * ss)
    hsic += torch.mean(zz) * torch.mean(ss)
    hsic -= 2 * torch.mean(torch.mean(zz, 1) * torch.mean(ss, 1))
    return torch.sqrt(torch.clamp(hsic, min=0))


def log_bernoulli_with_logits(x, logits, eps=1e-8, axis=-1):
    if eps > 0.0:
        logits = torch.clamp(logits, -18, 18)
    return -torch.sum(
        sigmoid_cross_entropy_with_logits(logits, x), axis)


def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    Note: All inputs are torch Tensors
    log likelihood (scalar) of a minibatch according to a zinb model.
    Notes:
    We parametrize the bernoulli using the logits, hence the softplus functions appearing

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
    eps: numerical stability constant
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    case_zero = (F.softplus((- pi + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps)))
                 - F.softplus(-pi))

    case_non_zero = - pi - F.softplus(-pi) + theta * torch.log(theta + eps) - theta * torch.log(
        theta + mu + eps) + x * torch.log(mu + eps) - x * torch.log(theta + mu + eps) + torch.lgamma(
        x + theta + eps) - torch.lgamma(theta + eps) - torch.lgamma(x + 1)

    res = torch.mul((x < eps).type(torch.float32), case_zero) + torch.mul((x > eps).type(torch.float32), case_non_zero)
    return torch.sum(res, dim=-1)


def log_nb_positive(x, mu, theta, eps=1e-8):
    """
    Note: All inputs should be torch Tensors
    log likelihood (scalar) of a minibatch according to a nb model.

    Variables:
    mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps: numerical stability constant
    """
    res = theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps) + x * torch.log(
        mu + eps) - x * torch.log(theta + mu + eps) + torch.lgamma(x + theta) - torch.lgamma(
        theta.view(1, theta.size(0))) - torch.lgamma(
        x + 1)
    return torch.sum(res)
