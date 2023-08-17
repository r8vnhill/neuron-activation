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


def tanh(x: torch.Tensor, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the tanh activation function for a given tensor.

    The tanh function is defined as:
    \[ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \]

    Its gradient is:
    \[ \text{tanh}'(x) = 1 - \text{tanh}^2(x) \]

    :param x: Input tensor for which to compute the tanh.
    :param gradient: If True, compute the gradient of the tanh with respect to its input.
    :return: Tensor containing the tanh values or its gradient.
    """
    if gradient:
        return 1 - tanh(x) ** 2
    exp_x = torch.exp(x)
    exp_neg_x = torch.exp(-1 * x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


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
        return (x >= 0).float()
    return torch.max(torch.zeros_like(x), x)


def swish(x: torch.Tensor, beta: float = 1.0, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the Swish activation function for a given tensor.

    The Swish function is defined as:
    \[ \text{swish}(x) = x \times \text{sigmoid}(\beta x) \]

    Its gradient is:
    \[ \text{swish}'(x) = \text{sigmoid}(\beta x) + \beta x \text{sigmoid}(\beta x) (1 - \text{sigmoid}(\beta x)) \]

    :param x: Input tensor for which to compute the Swish.
    :param beta: The beta value to use in the Swish function.
    :param gradient: If True, compute the gradient of the Swish with respect to its input.
    :return: Tensor containing the Swish values or its gradient.
    """
    if gradient:
        s = sigmoid(beta * x)
        return s + beta * x * s * (1 - s)
    return x * sigmoid(beta * x)


def celu(x: torch.Tensor, alpha: float = 1.0, gradient: bool = False) -> torch.Tensor:
    r"""
    Computes the CELU (Continuously Differentiable Exponential Linear Unit) activation function for a given tensor.

    The CELU function is defined as:
    \[ \text{celu}(x, \alpha) =
    \begin{cases}
    x & \text{if } x \geq 0 \\
    \alpha (\exp(\frac{x}{\alpha}) - 1) & \text{otherwise}
    \end{cases}
    \]

    Its gradient is:
    \[
    \frac{\partial\ \text{celu}(x, \alpha)}{\partial x} =
    \begin{cases}
    1 & \text{if } x \geq 0 \\
    \frac{\text{celu}(x, \alpha) - x e^{\frac{x}{\alpha}}}{\alpha} & \text{if } x < 0
    \end{cases}
    \]

    :param x: Input tensor for which to compute the CELU.
    :param alpha: The alpha value to use in the CELU function.
    :param gradient: If True, compute the gradient of the CELU with respect to its input.
    :return: Tensor containing the CELU values or its gradient.
    """
    zeros = torch.zeros_like(x)
    x_div_alpha = x / alpha

    if gradient:
        e = torch.exp(x_div_alpha)
        d_dx = torch.ones_like(x)
        d_dx[x < 0] = (celu(x, alpha)[x < 0] - x[x < 0] * e[x < 0]) / alpha
        return d_dx

    return torch.max(zeros, x) + torch.min(zeros, alpha * x_div_alpha.expm1())


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
