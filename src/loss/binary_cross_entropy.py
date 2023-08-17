import torch


def binary_cross_entropy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stable: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    Computes the binary cross-entropy loss between predictions and targets.

    For binary classification, where \( p(x) \) is the true label and \( q(x) \) is the
    predicted probability:

    .. math::
        H(p, q) = - \sum_{x \in X} p(x) log(q(x)) + (1 - p(x)) log(1 - q(x))

    :param predictions: The predictions of the network.
    :param targets: The target distribution.
    :param stable: Whether to use a stable implementation of the cross-entropy.
    :param eps: The epsilon value to use in the stable implementation.
    :return: The binary cross-entropy loss between predictions and targets.
    """
    predictions = (
        torch.clamp(predictions, min=eps, max=1 - eps) if stable else predictions
    )
    return -torch.mean(
        targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions)
    )
