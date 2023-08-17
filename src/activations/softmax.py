import torch


def softmax(tensor: torch.Tensor, dim: int, stable: bool = True) -> torch.Tensor:
    r"""
    Computes the softmax activation function for a given tensor.

    The softmax function is defined as:

    .. math::
        \mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}

    :param tensor: Input tensor for which to compute the softmax.
    :param dim: The dimension along which to compute the softmax.
    :param stable: Whether to use the stable version of softmax.
    :return: Tensor with the softmax values corresponding to each element of the input.
    """
    exp_tensor = torch.exp(
        tensor - torch.max(tensor, dim=dim, keepdim=True)[0]
        if stable
        else torch.zeros_like(tensor)
    )
    return exp_tensor / torch.sum(exp_tensor, dim=dim, keepdim=True)