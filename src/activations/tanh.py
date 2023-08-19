import torch


def tanh(x: torch.Tensor, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the tanh activation function for a given tensor.

    The tanh function is defined as:

    .. math::
        \mathrm{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Its gradient is:
        \mathrm{tanh}'(x) = 1 - \mathrm{tanh}^2(x)

    :param x: Input tensor for which to compute the tanh.
    :param gradient: If True, compute the gradient of the tanh with respect to its input.
    :return: Tensor containing the tanh values or its gradient.
    """
    if gradient:
        return 1 - tanh(x) ** 2
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-1 * x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
