import abc
from typing import Protocol

import torch
from torch import nn


class InputSupport(Protocol):
    """
    Protocol for classes that support input size.
    """

    # noinspection PyPropertyDefinition
    @property
    def input_size(self) -> int:
        """
        Gets the size of the input layer.
        """
        ...


class NeuralNetwork(nn.Module, InputSupport, abc.ABC):
    """
    Abstract base class for neural networks.
    """

    cache: list[torch.Tensor]

    @abc.abstractmethod
    def load_parameters(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        output_weights: torch.Tensor,
        output_biases: torch.Tensor,
    ):
        """
        Loads the provided parameters into the network.

        :param weights: A list of weight tensors for each layer.
        :param biases: A list of bias tensors for each layer.
        :param output_weights: The weight tensor for the output layer.
        :param output_biases: The bias tensor for the output layer.
        """
