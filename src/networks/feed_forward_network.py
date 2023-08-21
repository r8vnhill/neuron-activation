from typing import Callable, Optional

import torch
from torch import nn

from activations import softmax
from networks import NeuralNetwork


def get_init_weights(shape: tuple[int, int]) -> nn.Parameter:
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
    """
    Implements a feed-forward neural network (FFNN).
    The FFNN is a type of artificial neural network where the connections and data flow
    are unidirectional, moving from the input layer through any hidden layers and finally
    to the output layer.
    It's commonly used for tasks like regression, classification, and function
    approximation.

    The FFNN is characterized by its structure (layers of neurons, where each layer is
    fully connected to the next) and the activation functions applied to each neuron's
    output.

    :param n_features:
        The number of input features that the network expects. This
        corresponds to the size of the input layer.
    :param hidden_layer_sizes:
        A list containing the number of neurons for each hidden
        layer. For instance, [50, 30] would mean there are two hidden layers: the first
        with 50 neurons and the second with 30 neurons.
    :param activation_functions:
        A list of activation functions to be applied to the output of each layer (except
        the output layer). Activation functions introduce non-linearities that enable the
        network to capture more complex patterns. Examples of activation functions are
        ReLU, Sigmoid, and Tanh. The number of activation functions in the list should
        match the total number of layers (hidden).
    :param n_classes:
        The number of output classes or categories. In the context of classification, this
        would represent the number of possible labels. For regression, this would
        typically be 1.
    :param activation_function_parameters:
        A list containing parameters for each activation function, if they require any.
        For instance, the Leaky ReLU activation function has a parameter that defines the
        slope for values less than zero. If an activation function doesn't require
        parameters, its corresponding position in the list should be None. Default is None
        for all.
    """

    __weights: nn.ParameterList
    __biases: nn.ParameterList
    __activation_functions: list[Callable]
    __activation_functions_parameters_mask: list[Optional[nn.Parameter]]
    __activation_function_parameters: list[nn.Parameter]
    __cache: list[torch.Tensor]

    def __init__(
        self,
        n_features: int,
        hidden_layer_sizes: list[int],
        activation_functions: list[Callable],
        n_classes: int,
        activation_function_parameters: Optional[list[Optional[float]]] = None,
    ):
        super(FeedForwardNetwork, self).__init__()
        layer_sizes = [n_features] + hidden_layer_sizes + [n_classes]
        # Initialize the weights of the neural network using predefined initialization
        # function
        self.__weights = nn.ParameterList(
            [
                get_init_weights((layer_sizes[i], layer_sizes[i + 1]))
                for i in range(len(layer_sizes) - 1)
            ]
        )
        # Initialize biases for each layer, except the input layer
        self.__biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(size)) for size in layer_sizes[1:]]
        )
        self.__activation_functions = activation_functions

        # Handle optional activation function parameters
        # If parameters are provided, create a list of torch parameters; otherwise, fill
        # with None
        if activation_function_parameters:
            self.__activation_functions_parameters_mask = [
                nn.Parameter(torch.tensor(p)) if p else None
                for p in activation_function_parameters
            ]
        else:
            self.__activation_functions_parameters_mask = [
                None for _ in activation_functions
            ]
        self.__activation_function_parameters = [
            param
            for param in self.__activation_functions_parameters_mask
            if param is not None
        ]

    # Documented on NeuralNetwork
    @property
    def input_size(self):
        return self.__weights[0].shape[0]

    # Documented on NeuralNetwork
    def load_parameters(
        self,
        hidden_layer_weights: torch.Tensor,
        output_layer_weights: torch.Tensor,
        hidden_layer_biases: torch.Tensor,
        output_layer_biases: torch.Tensor,
    ):
        self.__weights = nn.ParameterList(
            [nn.Parameter(w) for w in hidden_layer_weights + [output_layer_weights]]
        )
        self.__biases = nn.ParameterList(
            [nn.Parameter(b) for b in hidden_layer_biases + [output_layer_biases]]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the neural network.

        In a feed-forward neural network, data flows in one direction: from input to
        output.
        During the forward pass, the input features are successively transformed by each
        layer of the network, making use of weights, biases, and activation functions.
        The result of this method is the network's prediction for the given input
        features.

        Algorithm:
        -----------
        1. Start with the input features.
        2. For each layer (excluding the last one):
            a. Compute the weighted sum of the inputs for each neuron. This is done by
               matrix multiplication (@ in Python) of the input features with the layer's
               weights, and then adding the bias.
            b. Apply the activation function to the result from step a.
        3. For the last layer, compute the weighted sum just like in step 2a.
        4. Apply the softmax function to the result from step 3 to get the network's
           probability distribution over the output classes.

        Mathematical Principles:
        ------------------------
        Given an input vector X and considering a single layer with weights W and bias b,
        the output O of that layer before applying the activation function is:

        O = XW + b

        After applying the activation function f, the final output of the layer becomes:

        O' = f(O)

        The above transformations are repeated for each layer in the network.

        The softmax function in the last step ensures that the outputs can be interpreted
        as probabilities, as it transforms its inputs into a probability distribution.

        :param features: torch.Tensor
            The input features to the network. Typically, a 2D tensor where each row is a
            sample and each column is a feature.
        :return: torch.Tensor
            The output of the network. For classification tasks, this is a probability
            distribution over the output classes.
        """
        self.__cache = []
        for weight, bias, activation, parameters in zip(
            self.__weights[:-1],
            self.__biases[:-1],
            self.__activation_functions,
            self.__activation_functions_parameters_mask,
        ):
            features = features @ weight + bias
            self.__cache.append(features)
            features = (
                activation(features, parameters.item())
                if parameters
                else activation(features)
            )
        return softmax(features @ self.__weights[-1] + self.__biases[-1], dim=1)

    def backward(
        self, features: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
    ):
        # The gradient of the loss with respect to the output of the network.
        # We compute this by finding the difference between the predicted values
        # and the actual targets, normalized by the number of samples.
        current_grad = (predictions - targets) / targets.size(0)
        # Iterate through each layer in reverse order, starting from the output layer
        # and moving toward the input layer.
        for i in range(len(self.__weights) - 1, 0, -1):
            # If there are no parameters associated with the activation function:
            if self.__activation_functions_parameters_mask[i - 1] is None:
                # Compute the gradient of the loss with respect to the weights.
                self.__weights[i].grad = (
                    self.__activation_functions[i - 1](self.__cache[i - 1]).t()
                    @ current_grad
                )
            else:
                # Compute the gradient of the loss with respect to the weights,
                # considering the parameters of the activation function.
                self.__weights[i].grad = (
                    self.__activation_functions[i - 1](
                        self.__cache[i - 1],
                        self.__activation_functions_parameters_mask[i - 1].item(),
                    ).t()
                    @ current_grad
                )
            # Compute the gradient of the loss with respect to the biases.
            self.__biases[i].grad = current_grad.sum(dim=0)
            # Compute the gradient of the loss with respect to the outputs of the previous
            # layer.
            h_grad = current_grad @ self.__weights[i].t()
            # Check if there are parameters associated with the activation function.
            if self.__activation_functions_parameters_mask[i - 1] is None:
                # Update the current gradient considering the gradient of the activation
                # function.
                current_grad = (
                    self.__activation_functions[i - 1](self.__cache[i - 1], gradient=True)
                    * h_grad
                )
            else:
                # Update the current gradient considering the gradient of the activation
                # function and its parameters. Also, compute the gradient for the
                # parameters of the activation function.
                current_grad, p_grad = self.__activation_functions[i - 1](
                    self.__cache[i - 1],
                    self.__activation_functions_parameters_mask[i - 1],
                    gradient=True,
                )
                current_grad *= h_grad
                self.__activation_functions_parameters_mask[i - 1].grad = (
                    p_grad * h_grad
                ).sum()
        # Compute the gradient of the loss with respect to the weights of the first layer.
        self.__weights[0].grad = features.t() @ current_grad
        # Compute the gradient of the loss with respect to the biases of the first layer.
        self.__biases[0].grad = current_grad.sum(dim=0)

    def __str__(self):
        """Return a string representation of the network's parameters."""
        return "\n".join([f"{name}:\t{param}" for name, param in self.named_parameters()])
