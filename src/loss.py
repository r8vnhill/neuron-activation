import torch


def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the cross-entropy between two distributions.

    The cross-entropy between two distributions p and q is defined as:

    H(p, q) = - \sum_{x \in X} p(x) log(q(x))

    :param p: The first distribution.
    :param q: The second distribution.
    :return: The cross-entropy between p and q.
    """
    assert p.shape == q.shape, "p and q must have the same shape"
    return -1 * torch.sum(p * torch.log(q))


def cross_entropy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stable: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    Computes the cross-entropy loss between predictions and targets.

    The cross-entropy loss between two distributions p and q is defined as:

    H(p, q) = - \sum_{x \in X} p(x) log(q(x))

    :param predictions: The predictions of the network.
    :param targets: The target distribution.
    :param stable: Whether to use a stable implementation of the cross-entropy.
    :param eps: The epsilon value to use in the stable implementation.
    :return: The cross-entropy loss between predictions and targets.
    """
    torch.binary_cross_entropy_with_logits()
    if stable:
        return -1 * torch.sum(targets * torch.log(predictions + eps))
    else:
        return cross_entropy(targets, predictions)
