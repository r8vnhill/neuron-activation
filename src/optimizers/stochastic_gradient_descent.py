from copy import deepcopy
from typing import Iterator

from torch import nn


class StochasticGradientDescent:
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    SGD is a simple yet very efficient approach to fitting linear classifiers and
    regressors under convex loss functions such as (linear) Support Vector Machines and
    Logistic Regression.
    Even though SGD has been around in the machine learning community for a long time,
    it has received a considerable amount of attention just recently in the context
    of large-scale learning.

    SGD updates parameters in the negative gradient direction to minimize the objective
    function.
    In essence, the gradient indicates the direction of the steepest ascent, and
    subtracting it steers the optimizer towards the local minimum.

    :param parameters: An iterator containing the parameters of the model.
    :param learning_rate: A hyperparameter determining the step size at each iteration
                          while moving toward a minimum of a loss function.
    """

    def __init__(self, parameters: Iterator[nn.Parameter], learning_rate: float):
        self.__parameters = parameters
        self.__learning_rate = learning_rate

    def step(self) -> None:
        """
        Performs a single optimization step.

        This function updates the parameters by subtracting the product of the learning
        rate and the gradient from the current parameters.
        This step directs the parameters towards the local minimum of the loss function.
        """
        for param in self.__parameters:
            param.data -= self.__learning_rate * param.grad

    @property
    def learning_rate(self) -> float:
        """
        Returns the learning rate of the optimizer.

        :return: Learning rate.
        """
        return self.__learning_rate

    @property
    def parameters(self) -> Iterator[nn.Parameter]:
        """
        Returns a deep copy of the model's parameters.

        This ensures that the original parameters are not accidentally modified,
        preserving immutability.

        :return: A deep copy of the model parameters.
        """
        return deepcopy(self.__parameters)


SGD = StochasticGradientDescent
