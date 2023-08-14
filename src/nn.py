import random
from typing import Callable

import torch
from matplotlib.pyplot import subplots
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from activation import softmax


class FeedForwardNetwork(nn.Module):
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
    activation_parameters: nn.ParameterList

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

    def forward(self, input_features: torch.tensor):
        """
        Computes the forward pass of the network given the input features.

        :param input_features: Tensor of input features.
        :return: Output tensor after the forward pass.
        """
        layer_features = input_features
        for weight, bias, activation, param in zip(
            self.weights, self.biases, self.activations, self.activation_parameters
        ):
            layer_features = activation(layer_features @ weight + bias, param.item())
        return softmax(layer_features, dim=1)

    def __str__(self):
        """
        Returns a string representation of the network.
        """
        return "\n".join([f"{name}:\t{param}" for name, param in self.named_parameters()])


def main():
    """
    Downloads the MNIST dataset and visualizes a random sample of images.

    This function performs the following tasks:
    1. Downloads the MNIST dataset and applies a transformation to convert images into tensors.
    2. Randomly selects `n_examples` images from the dataset.
    3. Plots the selected images with their respective class labels.

    :note: The images are displayed using matplotlib and have a resolution of 28x28 pixels.
    """
    dataset = MNIST(root="data/", download=True, transform=ToTensor())
    n_examples = 3
    fig, axes = subplots(nrows=n_examples, figsize=(2, n_examples * 3))
    for i in range(n_examples):
        idx = random.randint(0, len(dataset))
        image, label = dataset[idx]
        view = image.view(28, 28).numpy()
        axes[i].set_title(f"Class: {label}")
        axes[i].imshow(view)
    fig.show()


if __name__ == "__main__":
    main()
