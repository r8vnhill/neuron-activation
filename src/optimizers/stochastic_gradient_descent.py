from copy import deepcopy
from typing import Iterator, Callable, Optional

from torch import nn
from torch.optim import Optimizer


class StochasticGradientDescent:
    def __init__(self, parameters, learning_rate):
        self.__parameters = [p for p in parameters if p is not None]
        self.__learning_rate = learning_rate

    def step(self):
        for param in self.__parameters:
            param.data -= param.grad * self.__learning_rate

    @property
    def learning_rate(self) -> float:
        """
        Returns the learning rate of the optimizer.

        :return: Learning rate.
        """
        return self.__learning_rate

    @property
    def parameters(self) -> list[nn.Parameter]:
        """
        Returns a deep copy of the model's parameters.

        This ensures that the original parameters are not accidentally modified,
        preserving immutability.

        :return: A deep copy of the model parameters.
        """
        return deepcopy(self.__parameters)


SGD = StochasticGradientDescent
