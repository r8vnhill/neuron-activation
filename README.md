# PyTorch Playground

Welcome to the PyTorch Playground, a repository that offers implementations of common
neural network components using PyTorch.
Whether you are a beginner looking to explore the internals of various activations or an
experienced practitioner looking to play around with a custom feed-forward network, this
repository has you covered.

## Table of Contents

1. [Activations](#activations)
2. [FeedForward Network](#feedforward-network)
3. [MNIST Model](#mnist-visualization)
4. [Installation and Usage](#installation-and-usage)
5. [Contributions](#contributions)

## Activations

Located in the file named `activations`, this module offers various activation functions 
commonly used in neural networks.
These functions include:

- **Sigmoid**: `sigmoid(x)`
- **Hyperbolic Tangent (Tanh)**: `tanh(x)`
- **Rectified Linear Unit (ReLU)**: `relu(x)`
- **Swish**: `swish(x, beta)`
- **Continuously Differentiable Exponential Linear Units (CELU)**: `celu(x, alpha)`
- **Softmax**: `softmax(tensor, dim, stable)`

Each function is well documented, providing insights into its mathematical formulation and 
its PyTorch implementation.

## FeedForward Network

The file named `nn` contains the `FeedForwardNetwork` class, a simple and customizable 
feed-forward neural network.
It allows for:

- Custom layer sizes.
- Custom activation functions.
- Parameterized activations.

The `FeedForwardNetwork` is versatile, and you can load custom weights, retrieve the size 
of the input layer, and even get a string representation of the entire network.

## MNIST Model

Also located in the `nn` file is a utility function `main()`.
When run, this function:

- Downloads the MNIST dataset.
- Transforms the images into tensors.
- Randomly selects two images from the dataset.
- Creates a `FeedForwardNetwork` with 784 input neurons, two 16-neuron hidden layers, and
  10 output neurons.
- Loads the weights of a pre-trained model.
- Feeds the two images through the network.
- Displays the images and the network's predictions.

### Example Output

![MNIST Model Output](results/mnist.png)

## Installation and Usage

To use this repository:

1. Clone the repo:
    ```
    git clone https://github.com/r8vnhill/pytorch-playground.git
    ```
2. Install the necessary libraries:
    ```
    pip install -r requirements.txt
    ```
3. Run the desired modules or functions!


## Contributions

Feel free to fork this repository, submit pull requests, or create issues if you have 
suggestions or improvements!
