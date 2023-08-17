import torch

from activations import sigmoid


def swish(x: torch.Tensor, beta: float = 1.0, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the Swish activation function for a given tensor.

    The Swish function is defined as:

    .. math::
        \mathrm{swish}(x) = x \times \mathrm{sigmoid}(\beta x)

    Its gradient is:

    .. math::
        \mathrm{swish}'(x) =
            \mathrm{sigmoid}(\beta x)
            + \beta x \times \mathrm{sigmoid}(\beta x)
                \times (1 - \mathrm{sigmoid}(\beta x))


    :param x: Input tensor for which to compute the Swish.
    :param beta: The beta value to use in the Swish function.
    :param gradient: If True, compute the gradient of the Swish with respect to its input.
    :return: Tensor containing the Swish values or its gradient.
    """
    if gradient:
        s = sigmoid(beta * x)
        return s + beta * x * s * (1 - s)
    return x * sigmoid(beta * x)
