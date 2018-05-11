"""File for computing log likelihood of the data"""

import torch

from scvi.utils import to_cuda, no_grad, eval_modules


@no_grad()
@eval_modules()
def compute_log_likelihood(vae, data_loader):
    # Iterate once over the data_loader and computes the total log_likelihood
    log_lkl = 0
    for i_batch, tensors in enumerate(data_loader):
        if vae.use_cuda:
            tensors = to_cuda(tensors)
        sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors
        sample_batch = sample_batch.type(torch.float32)
        reconst_loss, kl_divergence = vae(sample_batch, local_l_mean, local_l_var, batch_index=batch_index,
                                          y=labels)
        log_lkl += torch.sum(reconst_loss).item()
    n_samples = len(data_loader.dataset) if data_loader.sampler is None else len(data_loader.sampler.indices)
    return log_lkl / n_samples


@no_grad()
@eval_modules()
def compute_true_log_likelihood(vae, data_loader, n_samples=100):
    # Compute log likelihood using importance sampling - must be defined in vae object
    log_lkl = 0
    for i_batch, tensors in enumerate(data_loader):
        if vae.use_cuda:
            tensors = to_cuda(tensors)
        sample_batch, local_l_mean, local_l_var, batch_index, labels = tensors
        sample_batch = sample_batch.type(torch.float32)
        reconst_loss = vae.log_likelihood(sample_batch, local_l_mean, local_l_var, batch_index=batch_index,
                                          labels=labels, n_samples=n_samples)
        log_lkl += torch.sum(reconst_loss).item()
    return log_lkl / len(data_loader.sampler.indices)


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

    def softplus(x):
        return torch.log(1 + torch.exp(x))

    case_zero = (softplus((- pi + theta * torch.log(theta + eps) - theta * torch.log(theta + mu + eps)))
                 - softplus(-pi))

    case_non_zero = - pi - softplus(-pi) + theta * torch.log(theta + eps) - theta * torch.log(
        theta + mu + eps) + x * torch.log(mu + eps) - x * torch.log(theta + mu + eps) + torch.lgamma(
        x + theta) - torch.lgamma(theta.view(1, theta.size(0))) - torch.lgamma(x + 1)

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
