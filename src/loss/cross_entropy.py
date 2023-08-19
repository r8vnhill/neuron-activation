import torch


def cross_entropy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stable: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    Computes the cross-entropy loss between predictions and targets.

    The cross-entropy loss between two distributions p and q is defined as:

    .. math::
        H(p, q) = - \sum_{x \in X} p(x) log(q(x))

    :param predictions: The predictions of the network.
    :param targets: The target distribution.
    :param stable: Whether to use a stable implementation of the cross-entropy.
    :param eps: The epsilon value to use in the stable implementation.
    :return: The cross-entropy loss between predictions and targets.
    """
    predictions = (
        torch.clamp(predictions, min=eps, max=1 - eps) if stable else predictions
    )
    return -torch.mean(targets * torch.log(predictions))
