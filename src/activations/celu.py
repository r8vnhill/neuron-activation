import torch


def celu(
    x: torch.Tensor, alpha: float = 1.0, gradient: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes the CELU (Continuously Differentiable Exponential Linear Unit) activation
    function for a given tensor.

    The CELU function is defined as:

    .. math::
        \mathrm{celu}(x, \alpha) = \begin{cases}
            x                                   & \text{if } x \geq 0 \\
            \alpha (\exp(\frac{x}{\alpha}) - 1) & \text{otherwise}
        \end{cases}

    Its gradient is:

    .. math::
        \frac{\partial\ \text{celu}(x, \alpha)}{\partial x} = \begin{cases}
            1 & \text{if } x \geq 0 \\
            \frac{\text{celu}(x, \alpha) - x e^{\frac{x}{\alpha}}}{\alpha}
                & \text{if } x < 0
        \end{cases}


    :param x: Input tensor for which to compute the CELU.
    :param alpha: The alpha value to use in the CELU function.
    :param gradient: If True, compute the gradient of the CELU with respect to its input.
    :return: Tensor containing the CELU values or its gradient.
    """
    if alpha == 0:
        raise ValueError("alpha cannot be 0")

    zeros = torch.zeros_like(x)
    x_div_alpha = x / alpha

    if gradient:
        e = x_div_alpha.exp()
        d_dx = torch.ones_like(x)
        d_dx[x < 0] = e[x < 0]
        zeros[x < 0] = (celu(x)[x < 0] - x[x < 0] * e[x < 0]) / alpha
        return d_dx, zeros  # d_dx, d_da

    return torch.max(zeros, x) + torch.min(zeros, alpha * x_div_alpha.expm1())
