from typing import Callable, Optional

import torch
from torch import nn

from activation import softmax
from . import NeuralNetwork


class FeedForwardNetwork(NeuralNetwork):
    """
    Implementation of a feed-forward neural network with customizable layer sizes and
    activations.

    This class provides an interface to create and utilize a simple feed-forward
    neural network with any given number of hidden layers, layer sizes, and activation
    functions.
    Activation functions can be parameterized, and those parameters can be learned during
    training.

    :ivar weights: A list of weight tensors for each layer.
    :ivar biases: A list of bias tensors for each layer.
    :ivar activations: A list of callable activation functions for each layer.
    :ivar activation_parameters: Optional parameters for the activation functions.

    :param input_size: Size of the input layer.
    :param hidden_sizes: List of sizes for the hidden layers.
    :param activations: List of callable activation functions for each layer.
    :param output_size: Size of the output layer.
    :param *args: Optional parameters for the activation functions.
    """

    weights: nn.ParameterList
    biases: nn.ParameterList
    activations: list[Callable[[torch.Tensor, ...], torch.Tensor]]
    activation_parameters: Optional[nn.ParameterList]

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        activations: list[Callable[[torch.Tensor, ...], torch.Tensor]],
        output_size: int,
        *args,
    ):
        """Initializes the FeedForwardNetwork with the provided parameters."""
        super(FeedForwardNetwork, self).__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(sizes[i], sizes[i + 1]))
                for i in range(len(sizes) - 1)
            ]
        )
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(h)) for h in sizes[1:]])
        self.activations = activations
        if args is not None and len(args) > 0:
            self.activation_parameters = nn.ParameterList(
                [nn.Parameter(param) for param in args]
            )
        else:
            self.activation_parameters = None

    @property
    def input_size(self) -> int:
        """
        Gets the size of the input layer.
        """
        return self.weights[0].shape[0]

    def load_parameters(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        output_weights: torch.Tensor,
        output_biases: torch.Tensor,
    ):
        """
        Load new weights and biases into the network.

        :param weights: List of weight tensors for each hidden layer.
        :param biases: List of bias tensors for each hidden layer.
        :param output_weights: Weight tensor for the output layer.
        :param output_biases: Bias tensor for the output layer.
        """
        self.weights = nn.ParameterList(
            [nn.Parameter(w) for w in weights + [output_weights]]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(b) for b in biases + [output_biases]]
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward propagation through the network using the provided input
        features.

        The forward pass sequentially applies weights, biases, and activation functions
        for each layer in the network. The output of the last layer is then passed through
        a softmax function to produce a probability distribution over classes.

        :param input_features: The input tensor containing the features for the network.
            Expected shape is (batch_size, input_size).
        :return: The output tensor representing the probability distribution over classes.
            Shape: (batch_size, output_size).
        """
        layer_features = input_features
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            layer_features = (
                activation(layer_features @ weight + bias, self.activation_parameters)
                if self.activation_parameters is not None
                else activation(layer_features @ weight + bias)
            )
        return softmax(layer_features @ self.weights[-1] + self.biases[-1], dim=1)

    def __str__(self):
        """
        Returns a string representation of the network.
        """
        return "\n".join([f"{name}:\t{param}" for name, param in self.named_parameters()])
