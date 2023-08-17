from typing import Optional

import torch
from torch import nn

from activations import softmax
from network import NeuralNetwork


def get_init_weights(shape: torch.Size) -> nn.Parameter:
    """
    Initialize weights for a neural network layer with random values from a normal
    distribution.

    This function initializes weights by sampling values from a standard normal
    distribution.
    The initialized weights are then wrapped in a `nn.Parameter`, making them trainable
    during the optimization process in PyTorch.

    :param shape: The shape of the weight tensor, typically (input_features,
                  output_features) for a fully connected layer.

    :return: A tensor of the provided shape with values initialized from a standard normal
             distribution and wrapped in a `nn.Parameter` for training.
    """
    weights = torch.randn(shape)
    return nn.Parameter(weights)


class FeedForwardNetwork(NeuralNetwork):
    def __init__(
        self,
        n_features: int,
        hidden_layer_sizes: list[int],
        activation_functions: list[callable],
        n_classes: int,
        activation_parameters: Optional[list[object]] = None,
    ):
        super(FeedForwardNetwork, self).__init__()

        sizes = [n_features] + hidden_layer_sizes + [n_classes]
        self.Ws = nn.ParameterList(
            [get_init_weights((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        )
        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(h)) for h in sizes[1:]])
        self.fs = activation_functions
        if activation_parameters is not None:
            self.fs_ps_mask = [
                nn.Parameter(torch.tensor(p)) if p else None
                for p in activation_parameters
            ]
        else:
            self.fs_ps_mask = [None for _ in activation_functions]
        self.fs_ps = nn.ParameterList([p for p in self.fs_ps_mask if p])

    @property
    def in_size(self):
        return self.Ws[0].shape[0]

    def load_parameters(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        output_weights: torch.Tensor,
        output_biases: torch.Tensor,
    ):
        self.Ws = nn.ParameterList([nn.Parameter(w) for w in weights] + [output_weights])
        self.bs = nn.ParameterList([nn.Parameter(b) for b in biases] + [output_biases])

    def resumen(self):
        # Usa self.parameters() o self.named_parameters().
        for name, p in self.named_parameters():
            print("{}:\t{}".format(name, p.size()))
        pass

    def forward(self, x):
        self.cacheU = []  # , self.cacheH = [], []
        for W, b, f, p in zip(self.Ws[:-1], self.bs[:-1], self.fs, self.fs_ps_mask):
            x = torch.mm(x, W) + b
            self.cacheU.append(x)
            x = f(x, p.item()) if p else f(x)
        #       self.cacheH.append(x)
        return softmax(torch.mm(x, self.Ws[-1]) + self.bs[-1], dim=1)

    # nuevo código Tarea 2
    def backward(self, x, y, y_pred):
        current_grad = (y_pred - y) / y.size(0)

        for i in range(len(self.Ws) - 1, 0, -1):
            if self.fs_ps_mask[i - 1] is None:
                self.Ws[i].grad = self.fs[i - 1](self.cacheU[i - 1]).t() @ current_grad
            else:
                self.Ws[i].grad = (
                    self.fs[i - 1](self.cacheU[i - 1], self.fs_ps_mask[i - 1].item()).t()
                    @ current_grad
                )
            self.bs[i].grad = current_grad.sum(dim=0)
            h_grad = current_grad @ self.Ws[i].t()

            if self.fs_ps_mask[i - 1] is None:
                current_grad = self.fs[i - 1](self.cacheU[i - 1], gradient=True) * h_grad
            else:
                current_grad, p_grad = self.fs[i - 1](
                    self.cacheU[i - 1], self.fs_ps_mask[i - 1], gradient=True
                )
                current_grad *= h_grad
                self.fs_ps_mask[i - 1].grad = (p_grad * h_grad).sum()

        self.Ws[0].grad = x.t() @ current_grad
        self.bs[0].grad = current_grad.sum(dim=0)

    def __str__(self):
        """Return a string representation of the network's parameters."""
        return "\n".join([f"{name}:\t{param}" for name, param in self.named_parameters()])
