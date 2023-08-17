import torch

import loss
from activations import sigmoid, relu
from networks import FeedForwardNetwork


def gradient_checking(epsilon: float = 1e-6):
    """
    Perform gradient checking to ensure the backpropagation implementation is correct.

    Gradient checking is a technique used to ensure that the gradients calculated by
    backpropagation are consistent with numerically approximated gradients.
    This function tests the gradient computations of a FeedForwardNetwork.

    Args:
        epsilon: A small value used for numerical gradient approximation.

    Note:
        This function should ONLY be used for debugging purposes during development,
        and not during actual training, due to its high computational cost.
    """

    # Ensure computations do not track history (for memory efficiency)
    with torch.no_grad():
        # Define some constants and random data
        n = 100  # Number of examples
        features = 300  # Number of input features
        classes = 10  # Number of output classes

        # Initialize a feed-forward neural network model
        model = FeedForwardNetwork(features, [100, 200], [sigmoid, relu], classes)
        params = list(model.parameters())

        # Generate random input data
        x = torch.randn(n, features)

        # Create one-hot encoded target labels
        y = torch.zeros(n, classes)
        targets = torch.randint(0, classes, (n,))
        y[torch.arange(n), targets] = 1

        # Iterate through each parameter tensor in the model for gradient checking
        for i in range(len(params)):
            # Perturb the parameter by -epsilon and calculate the loss
            params[i] -= epsilon
            y_pred = model(x)
            loss_1 = loss.cross_entropy(y_pred, y)
            print(f"Loss 1 (Negative perturbation): {loss_1}")

            # Perturb the parameter by +epsilon and calculate the loss
            params[i] += 2 * epsilon
            y_pred = model(x)
            loss_2 = loss.cross_entropy(y_pred, y)
            print(f"Loss 2 (Positive perturbation): {loss_2}")

            # Compute the approximate gradient using central difference
            approx_grad = (loss_2 - loss_1) / (2 * epsilon)
            print(f"Approximate gradient: {approx_grad}")

            # Reset parameter to its original state
            params[i] -= epsilon

            # Use the backward method to compute gradients
            y_pred = model(x)
            model.backward(x, y, y_pred)

            # Calculate and print the difference between the approximated and actual
            # gradient
            print(
                "Difference between approximated and actual gradient: "
                f"{torch.abs(approx_grad - torch.mean(params[i].grad))}"
            )


if __name__ == "__main__":
    gradient_checking()
