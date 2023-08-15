import abc
from typing import Protocol

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

    ...
