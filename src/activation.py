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


if __name__ == "__main__":
    print(sigmoid(torch.tensor(17.0)))
