from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer

from activations import softmax
from datasets import SizedDataset
from devices import Device
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
    Defines a FeedForward Neural Network, also known as a Multi-Layer Perceptron (MLP).

    FeedForward networks are the quintessential deep learning models where information
    moves in only one direction: forward.
    The data flows from the input layer, through the hidden layers, to the output layer
    without looping back.

    Attributes:
    - weights (nn.ParameterList): List of weight matrices for each layer.
    - biases (nn.ParameterList): List of bias vectors for each layer.
    - activation_functions (list[callable]): List of activation functions for each hidden
      layer.
    - activation_parameters_mask (list[Optional[torch.Tensor]]): List of optional
      parameters for the activation functions.
    - activation_parameters (nn.ParameterList): List of non-null parameters from the
      activation_parameters_mask.
    - cache (list[torch.Tensor]): A cache to store intermediate results, useful for
      backpropagation.

    Args:
    - n_features (int): Number of input features.
    - hidden_layer_sizes (list[int]): List containing the number of neurons for each
      hidden layer.
    - activation_functions (list[callable]): List of activation functions for each hidden
      layer.
    - n_classes (int): Number of output classes or units in the output layer.
    - activation_parameters (Optional[list[object]]): Optional list of parameters for the
      activation functions.
    """

    def __init__(
        self,
        n_features: int,
        hidden_layer_sizes: list[int],
        activation_functions: list[callable],
        n_classes: int,
        activation_parameters: Optional[list[object]] = None,
    ):
        super(FeedForwardNetwork, self).__init__()

        # Construct the sizes list which contains sizes for input, hidden layers, and
        # output.
        sizes = [n_features] + hidden_layer_sizes + [n_classes]

        # Initialize the weight matrices for each layer.
        self.weights = nn.ParameterList(
            [get_init_weights((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        )

        # Initialize the bias vectors for each layer.
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_size)) for input_size in sizes[1:]]
        )

        # Store the provided activation functions.
        self.activation_functions = activation_functions

        # If specific parameters for activation functions are provided, store them.
        if activation_parameters is not None:
            self.activation_parameters_mask = [
                nn.Parameter(torch.tensor(p)) if p else None
                for p in activation_parameters
            ]
        else:
            # If no specific parameters are provided, initialize the list with None.
            self.activation_parameters_mask = [None for _ in activation_functions]

        # Filter out the non-null parameters and store them in a ParameterList.
        self.activation_parameters = nn.ParameterList(
            [p for p in self.activation_parameters_mask if p]
        )

    # Documented in superclass
    @property
    def input_size(self):
        return self.weights[0].shape[0]

    # Documented in superclass
    def load_parameters(
        self,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
        output_weights: torch.Tensor,
        output_biases: torch.Tensor,
    ):
        self.weights = nn.ParameterList(
            [nn.Parameter(w) for w in weights] + [output_weights]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(b) for b in biases] + [output_biases]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the forward propagation through a feed-forward neural network.

        A feed-forward neural network consists of multiple layers, where each layer
        is composed of weights, biases, and activation functions.
        In this function, input data `x` is propagated through the network using the
        following process:

        1. The input data `x` is linearly transformed by the layer's weights and biases:
        `x = x*W + b`.
        2. The result is passed through an activation function to introduce non-linearity.
        3. Steps 1 and 2 are repeated for each layer in the network.
        4. The output of the final layer is passed through the softmax function to
           produce a probability distribution over the target classes.

        Mathematical principle:
        - Linear Transformation: `y = x*W + b`
          where `x` is the input data, `W` is the weights matrix, and `b` is the bias
          vector.
        - Activation Function: Introduces non-linearity into the network.
          Examples include Sigmoid, ReLU, Tanh, etc.
          The specific function and potential parameters are determined by `f` and `p`.
        - Softmax Function: Converts raw output scores (logits) to probabilities, such
          that they sum to 1.

        Formally, the forward propagation through a feed-forward neural network is defined
        as follows:

        .. math::
            \begin{aligned}
              h^{(\mathcal{l})} &= f^{(\mathcal{l})}(h^{(\mathcal{l}-1)}W^{(\mathcal{l})}
                + b^{(\mathcal{l})}) \\
              \hat{y} &= \mathrm{softmax}(h^{(L)}W^{(L+1)} + b^{(L+1)})
            \end{aligned}

        :param features: The input tensor containing data to be propagated through the
                         network.
                         Shape: [batch_size, input_features].
        :return: The output tensor after forward propagation. Represents a probability
                 distribution over target classes.
                 Shape: [batch_size, number_of_classes].

        Attributes:
            cache (list): List to store the linear transformations (before activation) of each layer.
            weights (list): List of weight tensors for each layer in the network.
            biases (list): List of bias tensors for each layer in the network.
            activation_functions (list): List of callable activation functions for each layer.
            activation_parameters_mask (list): List of potential parameters for each activation function.
                Could be None if the activation function doesn't require any parameters.
        """
        self.cache = []
        layer_input = features
        for weight, bias, activation, parameters in zip(
            self.weights[:-1],
            self.biases[:-1],
            self.activation_functions,
            self.activation_parameters_mask,
        ):
            layer_input = layer_input @ weight + bias
            self.cache.append(layer_input)
            layer_input = (
                activation(layer_input, parameters.item())
                if parameters
                else activation(layer_input)
            )
        return softmax(layer_input @ self.weights[-1] + self.biases[-1], dim=1)

    def backward(
        self, input_data: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
    ) -> None:
        r"""
        Computes the backward pass, updating gradients for weights and biases.

        The backpropagation algorithm is a supervised learning algorithm for multilayer
        feed-forward networks from the field of deep learning.
        It consists of:

        1. Performing a feedforward pass.
        2. Comparing the output of the model with the desired output.
        3. Computing the gradient of the error with respect to each parameter.
        4. Adjusting the parameters in the direction that reduces the error.

        Mathematical principles:
        - The starting point is the loss function's gradient, which measures how the
          output differs from the desired output.

          .. math::
            \frac{\partial L}{\partial y^{(L)}} = \frac{1}{N}(y^{(L)} - \hat{y})

        - For each layer, starting from the last hidden layer going backward:
          Using the chain rule, the error is propagated backward.
          The gradient with respect to the weights is computed by multiplying the gradient
          of the error at the output of the layer by the input to the layer:

          .. math::
            \frac{\partial L}{\partial W^{(\mathcal{l})}} = h^{(\mathcal{l}-1)T}
                \frac{\partial L}{\partial h^{(\mathcal{l})}}

        - The gradient with respect to the biases is just the sum of the gradients at the
          output of the layer.

        - To propagate the error backward to the previous layer:

          .. math::
            \frac{\partial L}{\partial h^{(\mathcal{l}-1)}} =
                \frac{\partial L}{\partial h^{(\mathcal{l})}}W^{(\mathcal{l})T}

        :param input_data: The input tensor data. Shape (N, input_features), where N is the batch size and input_features
                           is the number of input features.
        :param target: The ground truth tensor. Shape (N, output_features), where N is the batch size and output_features
                       is the number of output features (typically, the number of classes in classification tasks).
        :param prediction: The model's predictions based on forward propagation. Should have the same shape as target.
        """
        # Compute the gradient of the loss with respect to the predicted values.
        current_grad = (prediction - target) / target.size(0)
        # Iterate over each layer in reverse order to back-propagate the error.
        for i in range(len(self.weights) - 1, 0, -1):
            current_grad = self._compute_gradient_for_layer(i, current_grad)
        # Handle the output layer's gradient computation.
        self._compute_gradient_for_output_layer(input_data, current_grad)

    def _compute_gradient_for_layer(
        self, layer_idx: int, current_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient for a specific layer in the neural network.

        :param layer_idx: Index of the layer for which the gradient should be computed.
        :param current_grad: Current gradient tensor propagated from the subsequent layer.
        :return: Updated gradient for this layer to be used for previous layers.
        """
        # Check if there are any parameters for the activation function.
        mask = self.activation_parameters_mask[layer_idx - 1]
        # Get the activation function for this layer.
        activation = self.activation_functions[layer_idx - 1]
        # Gradient holder for the current layer.
        layer_grad = current_grad
        if mask is None:
            # If there are no parameters for the activation function, simply compute the
            # gradient.
            self.weights[layer_idx].grad = (
                activation(self.cache[layer_idx - 1]).T @ layer_grad
            )
        else:
            # If there are parameters for the activation function, consider them in the
            # gradient.
            # noinspection PyUnresolvedReferences
            self.weights[layer_idx].grad = (
                activation(self.cache[layer_idx - 1], mask.item()).T @ layer_grad
            )
        # Compute the gradient for the biases.
        self.biases[layer_idx].grad = torch.sum(layer_grad, dim=0)
        if mask is None:
            # Compute the gradient with respect to the previous layer.
            layer_grad = activation(self.cache[layer_idx - 1], gradient=True) * (
                layer_grad @ self.weights[layer_idx].T
            )
        else:
            # Compute the gradient with respect to activation parameters.
            layer_grad, param_grad = activation(
                self.cache[layer_idx - 1],
                mask,
                gradient=True,
            )
            layer_grad *= layer_grad @ self.weights[layer_idx].T
            mask.grad = torch.sum(param_grad * layer_grad)
        return layer_grad

    def _compute_gradient_for_output_layer(
        self, x: torch.Tensor, current_grad: torch.Tensor
    ) -> None:
        """
        Computes the gradient for the output layer.

        :param x: The input tensor data.
        :param current_grad: Current gradient tensor propagated from the previous hidden
                             layer.
        """
        # Update the gradients for the weights and biases of the output layer.
        self.weights[0].grad = x.T @ current_grad
        self.biases[0].grad = torch.sum(current_grad, dim=0)

    def __str__(self):
        """Return a string representation of the network's parameters."""
        return "\n".join([f"{name}:\t{param}" for name, param in self.named_parameters()])


def train_feed_forward_network(
    network: FeedForwardNetwork,
    dataset: SizedDataset,
    optimizer: Optimizer,
    epochs: int = 1,
    batch_size: int = 1,
    device: Device = Device.CPU,
) -> None:
    pass
