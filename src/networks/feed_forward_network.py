from typing import Callable, Optional

import torch
from torch import nn

from activations import softmax
from networks import NeuralNetwork


def _get_init_weights(shape: tuple[int, int]) -> nn.Parameter:
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
                _get_init_weights((layer_sizes[i], layer_sizes[i + 1]))
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
    ) -> None:
        """
        Computes the backward pass of the neural network using the back-propagation
        algorithm.

        Back-propagation is a supervised learning algorithm for training feedforward
        neural networks.
        It computes the gradient of the loss function with respect to each weight by
        propagating the gradient backward in the network. The main principle is the chain
        rule of calculus which allows us to find the gradient of the loss with respect to
        any weight in the network.

        Here's a brief overview of the steps taken in back-propagation:

        1. **Output Error Calculation**: Compute the difference between the network's
            prediction and the actual target. This difference, normalized by the number of
            samples, gives the gradient of the loss with respect to the outputs of the
            last layer.
        2. **Backward Pass Through Layers**: For each layer, starting from the last and
            moving to the first, compute the gradient of the loss with respect to the
            weights, biases, and outputs of that layer. This involves:
            - Using the gradient of the loss with respect to the outputs of the current
                layer to compute the gradient with respect to the weights and biases.
            - Using the gradient with respect to the weights and the outputs of the
                previous layer to compute the gradient with respect to the outputs of the
                previous layer.
        3. **Activation Function Gradient**: When computing the gradient with respect to
            the outputs of a layer, we consider the gradient of the layer's activation
            function. If the activation function has parameters, we also compute the
            gradient with respect to those parameters.

        :param features: The input samples to the neural network.
            Shape: [batch_size, n_features].
        :param targets: The actual target values corresponding to the input samples.
            Shape: [batch_size, n_classes].
        :param predictions: The predicted outputs produced by the forward pass of the
            neural network.
            Shape: [batch_size, n_classes].
        :return: None. The method updates the gradients for the weights and biases
            in-place.
        """
        current_grad = FeedForwardNetwork._initialize_gradient(predictions, targets)
        # Handle all layers except the first
        for i in range(len(self.__weights) - 1, 0, -1):
            current_grad = self._layer_gradient(i, current_grad)
        self._first_layer_gradient(features, current_grad)

    @staticmethod
    def _initialize_gradient(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Initializes the gradient for the backward pass of the neural network.

        This method calculates the gradient of the loss with respect to the output of the
        network, which forms the starting point for the back-propagation algorithm.
        Specifically, it computes the difference between the predicted values and the
        actual target values. This difference is then normalized by the number of samples
        to provide the average gradient.

        The method assumes that a cross-entropy loss or similar is used, where the
        derivative of the loss with respect to the predictions is simply
        `(predictions - targets)`.

        :param predictions: The predicted outputs produced by the forward pass of the
            neural network.
            Shape: [batch_size, n_classes].
        :param targets: The actual target values corresponding to the input samples.
            Shape: [batch_size, n_classes].
        :return: Tensor representing the gradient of the loss with respect to the
            network's outputs.
            Shape: [batch_size, n_classes].
        """
        return (predictions - targets) / targets.size(0)

    def _layer_gradient(self, i: int, current_grad: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient for the weights, biases, and outputs of a given layer
        during the back-propagation of the neural network.

        The method updates the gradient values for the weights and biases of the specified
        layer based on the input gradient (`current_grad`) which represents the gradient
        of the loss with respect to the outputs of the current layer. Additionally, it
        computes the gradient of the loss with respect to the outputs of the previous
        layer, which will be used as input for the next step in back-propagation.

        The gradient computation involves a few steps:
        1. The gradient for the weights is computed by taking the outer product of the
            gradient of the activation function of the layer's outputs (or their
            derivatives) and the current gradient.
        2. The gradient for the biases is the sum of the current gradient over the batch
            dimension.
        3. The gradient for the previous layer's outputs is computed by multiplying the
            current gradient with the transpose of the current layer's weights.

        :param i: Index of the current layer for which the gradient computations are being
            performed. 0-based index with 0 representing the first hidden layer.
        :param current_grad: Tensor representing the gradient of the loss with respect to
            the outputs of the current layer.
            Shape: [batch_size, n_neurons_in_current_layer].
        :return: Tensor representing the gradient of the loss with respect to the outputs
            of the previous layer.
            Shape: [batch_size, n_neurons_in_previous_layer].
        """
        if self.__activation_functions_parameters_mask[i - 1] is None:
            self.__weights[i].grad = (
                self.__activation_functions[i - 1](self.__cache[i - 1]).t() @ current_grad
            )
        else:
            self.__weights[i].grad = (
                self.__activation_functions[i - 1](
                    self.__cache[i - 1],
                    self.__activation_functions_parameters_mask[i - 1].item(),
                ).t()
                @ current_grad
            )
        self.__biases[i].grad = current_grad.sum(dim=0)
        h_grad = current_grad @ self.__weights[i].t()

        return self._activation_gradient(i, h_grad)

    def _activation_gradient(self, i: int, h_grad: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient of the loss with respect to the outputs of the (i-1)th
        layer, considering the activation function and its possible parameters.

        This method handles the derivative of the activation function. If the activation
        function has parameters, the gradient with respect to those parameters will also
        be computed.

        The gradient computation for the outputs of the previous layer (i.e., before
        activation) is done by element-wise multiplication of the derivative of the
        activation function applied to the outputs and the input gradient (`h_grad`).

        For activation functions with parameters:
        1. Compute the derivative of the activation function with respect to both its
            outputs and its parameters.
        2. Update the gradient of the activation function's parameter by summing the
            product of the computed parameter gradient (`p_grad`) and `h_grad`.
        3. Update the current gradient to reflect the combined effects of the gradient due
            to the outputs and the gradient due to the parameters.

        :param i: Index of the current layer. 0-based index, where 0 represents the first
            hidden layer.
        :param h_grad: Tensor representing the gradient of the loss with respect to the
            outputs (before activation) of the current layer.
            Shape: [batch_size, n_neurons_in_current_layer].
        :return: Tensor representing the gradient of the loss with respect to the outputs
            (before activation) of the previous layer.
            Shape: [batch_size, n_neurons_in_previous_layer].
        """
        if self.__activation_functions_parameters_mask[i - 1] is None:
            return (
                self.__activation_functions[i - 1](self.__cache[i - 1], gradient=True)
                * h_grad
            )
        else:
            current_grad, p_grad = self.__activation_functions[i - 1](
                self.__cache[i - 1],
                self.__activation_functions_parameters_mask[i - 1],
                gradient=True,
            )
            self.__activation_functions_parameters_mask[i - 1].grad = (
                p_grad * h_grad
            ).sum()
            return current_grad * h_grad

    def _first_layer_gradient(self, features: torch.Tensor, current_grad: torch.Tensor):
        """
        Computes the gradient of the loss with respect to the weights and biases of the
        first layer of the neural network.

        Backpropagation in deep neural networks is typically performed from the output
        layer backwards to the input layer. The gradient for the first layer is unique in
        that it depends directly on the input features rather than the outputs of a
        previous layer.
        This method handles this special case.

        To compute the weight gradient:
        1. The method calculates the outer product of the transposed input `features` and
           the `current_grad`, which represents the gradient of the loss with respect to
           the outputs (after activation) of the first layer.
        For the biases gradient:
        2. Since the bias term is added element-wise to every sample in the batch,
           the gradient for the bias is the sum of `current_grad` across all samples
           (i.e., along the batch dimension).

        :param features: Tensor representing the input features to the neural network.
            Shape: [batch_size, n_features].
        :param current_grad: Tensor representing the gradient of the loss with respect to
            the outputs (after activation) of the first layer.
            Shape: [batch_size, n_neurons_in_first_layer].
        """
        self.__weights[0].grad = features.t() @ current_grad
        self.__biases[0].grad = current_grad.sum(dim=0)

    def __str__(self):
        """Return a string representation of the network's parameters."""
        return "\n".join([f"{name}:\t{param}" for name, param in self.named_parameters()])
