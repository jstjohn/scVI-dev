import torch
from torch.distributions import register_kl, Multinomial


def iterate(obj, func):
    t = type(obj)
    if t is list or t is tuple:
        return t([iterate(o, func) for o in obj])
    else:
        return func(obj) if obj is not None else None


def broadcast_labels(y, *o, n_broadcast=-1):
    if not len(o):
        raise "Broadcast must have at least one reference argument"
    if y is None:
        ys = enumerate_discrete(o[0], n_broadcast)
        new_o = iterate(o, lambda x: x.repeat(n_broadcast, 1) if len(x.size()) == 2 else x.repeat(n_broadcast))
    else:
        ys = one_hot(y, n_broadcast)
        new_o = o
    return (ys,) + new_o


def repeat(*os, n_samples=10):
    return [o.repeat(1, n_samples).view(-1, o.size(1)) for o in os]


def one_hot(index, n_cat):
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def enumerate_discrete(x, y_dim):
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1, device=x.device, dtype=torch.long) * label)
        return one_hot(labels, y_dim)

    batch_size = x.size(0)
    return torch.cat([batch(batch_size, i) for i in range(y_dim)])


def log_mean_exp(x, axis):
    m = torch.max(x, dim=axis, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m), dim=axis, keepdim=True))


@register_kl(Multinomial, Multinomial)
def kl_multinomial_multinomial(p, q):
    return torch.sum(torch.mul(p.probs, torch.log(q.probs) - torch.log(p.probs + 1e-8)), dim=-1)
