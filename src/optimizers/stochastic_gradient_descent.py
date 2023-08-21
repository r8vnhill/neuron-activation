from copy import deepcopy

from torch import nn


class StochasticGradientDescent:
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    Stochastic Gradient Descent is a simple yet very efficient approach to
    discriminative learning of linear classifiers under convex loss functions
    such as linear SVM and logistic regression. It has received significant
    attention in machine-learning because of its efficiency and simplicity.

    This class provides methods to update model parameters using the calculated
    gradients and a fixed learning rate.

    :param parameters: Model parameters to be updated.
    :param learning_rate: Learning rate for the optimizer.
    """

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
