import pytest
import torch
from hypothesis import given, strategies, settings
from activation import sigmoid

# Disabling the hypothesis deadline for all tests
settings.register_profile("no_deadlines", deadline=None)
settings.load_profile("no_deadlines")


@given(sig_input=strategies.floats(allow_nan=False))
def test_sigmoid(sig_input):
    """
    Tests the correctness of the custom sigmoid activation function.

    Performs property-based testing on the custom sigmoid function using randomly
    generated float values. The function checks the following:
    - The output of the sigmoid function should always be <= 1.0.
    - The output of the sigmoid function should always be >= 0.0.
    - The output matches the result from PyTorch's built-in torch.sigmoid function.

    :param sig_input: The float input to test with the sigmoid function.
    """
    result = sigmoid(torch.tensor(sig_input))
    assert torch.all(torch.le(result, torch.tensor(1.0)))
    assert torch.all(torch.ge(result, torch.tensor(0.0)))
    assert torch.equal(result, torch.sigmoid(torch.tensor(sig_input)))


# Executes all tests in this module using pytest if run as the main script.
if __name__ == "__main__":
    pytest.main([__file__])
