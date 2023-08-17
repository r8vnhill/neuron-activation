import torch


def sigmoid(x: torch.Tensor, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the sigmoid activation function for a given tensor.

    The sigmoid function is defined as:
    \[ \text{sigmoid}(x) = \frac{1}{1 + e^{-x}} \]

    Its gradient is:
    \[ \text{sigmoid}'(x) = \text{sigmoid}(x)(1 - \text{sigmoid}(x)) \]

    :param x: Input tensor for which to compute the sigmoid.
    :param gradient: If True, compute the gradient of the sigmoid with respect to its input.
    :return: Tensor containing the sigmoid values or its gradient.
    """
    sigmoid_output = 1 / (1 + torch.exp(-x))
    if gradient:
        return sigmoid_output * (1 - sigmoid_output)
    return sigmoid_output