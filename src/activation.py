import torch


def sigmoid(x: torch.Tensor):
    r"""
    Computes the sigmoid activation function for a given tensor.

    The sigmoid function is defined as:

    sigmoid(x) = 1 / (1 + e^(-x))

    :param x: Input tensor for which to compute the sigmoid.
    :return: Tensor with the sigmoid values corresponding to each element of the input.
    """
    return 1 / (1 + torch.exp(-1 * x))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the tanh activation function for a given tensor.

    The tanh function is defined as:

    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    :param x: Input tensor for which to compute the tanh.
    :return: Tensor with the tanh values corresponding to each element of the input.
    """
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-1 * x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the ReLU activation function for a given tensor.

    The ReLU function is defined as:

    relu(x) = max(0, x)

    :param x: Input tensor for which to compute the ReLU.
    :return: Tensor with the ReLU values corresponding to each element of the input.
    """
    return torch.max(torch.zeros_like(x), x)


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Computes the Swish activation function for a given tensor.

    The Swish function is defined as:

    swish(x) = x * sigmoid(beta * x)

    :param x: Input tensor for which to compute the Swish.
    :param beta: The beta value to use in the Swish function.
    :return: Tensor with the Swish values corresponding to each element of the input.
    """
    return x * sigmoid(beta * x)


def celu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Computes the CELU activation function for a given tensor.

    The CELU function is defined as:

    celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))

    :param x: Input tensor for which to compute the CELU.
    :param alpha: The alpha value to use in the CELU function.
    :return: Tensor with the CELU values corresponding to each element of the input.
    """
    return relu(x) + torch.min(torch.zeros_like(x), alpha * (torch.exp(x / alpha) - 1))


def softmax(tensor: torch.Tensor, dim: int, stable: bool = True) -> torch.Tensor:
    """
    Computes the softmax activation function for a given tensor.

    The softmax function is defined as:

    softmax(x) = e^(x_i) / sum(e^(x_j)) for all j

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


if __name__ == "__main__":
    print(sigmoid(torch.tensor(17.0)))
    print(tanh(torch.tensor(2.0)))
    print(torch.tanh(torch.tensor(2.0)))
    print(softmax(torch.tensor([1.0, 2.0, 3.0]), 0))
    print(torch.softmax(torch.tensor([1.0, 2.0, 3.0]), 0))
