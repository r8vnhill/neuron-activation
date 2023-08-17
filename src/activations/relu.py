import torch


def relu(x: torch.Tensor, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the ReLU activation function for a given tensor.

    The ReLU function is defined as:
    \[ \text{relu}(x) = \max(0, x) \]

    Its gradient is:
    \[ \text{relu}'(x) =
    \begin{cases}
    1 & \text{if } x \geq 0 \\
    0 & \text{otherwise}
    \end{cases}
    \]

    :param x: Input tensor for which to compute the ReLU.
    :param gradient: If True, compute the gradient of the ReLU with respect to its input.
    :return: Tensor containing the ReLU values or its gradient.
    """
    if gradient:
        output = torch.zeros_like(x)
        output[x >= 0] = 1
        return output
    return torch.max(torch.zeros_like(x), x)
