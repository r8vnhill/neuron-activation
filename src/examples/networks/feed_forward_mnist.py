import random
from pathlib import Path

import torch
from matplotlib import pyplot
from matplotlib.pyplot import subplots
from numpy import loadtxt
from torch import nn
from torchvision.datasets import MNIST, VisionDataset
from torchvision.transforms import ToTensor

from activations import relu
from networks import FeedForwardNetwork, evaluate_network

ModelParameters = tuple[
    list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor
]


def load_parameters_from_files(data_path: Path) -> ModelParameters:
    """
    Load model parameters from the given path.

    :param data_path: The path where the parameter files are stored.

    :return: Tuple containing weights, biases, output_weights, and output_biases as
             tensors.
    """
    weights = [torch.from_numpy(loadtxt(data_path / f"W{i}.txt")).float() for i in (1, 2)]
    biases = [torch.from_numpy(loadtxt(data_path / f"b{i}.txt")).float() for i in (1, 2)]
    output_weights = torch.from_numpy(loadtxt(data_path / "U.txt")).float()
    output_biases = torch.from_numpy(loadtxt(data_path / "c.txt")).float()

    return weights, biases, output_weights, output_biases


def visualize_sample_predictions(
    dataset: VisionDataset, model: nn.Module, n_examples: int
) -> pyplot.Figure:
    """
    Visualizes a sample of images from the dataset along with their predicted class labels.

    :param dataset: The MNIST dataset to sample from.
    :param model: The trained model for predictions.
    :param n_examples: Number of examples to visualize.

    :return: Matplotlib figure containing the visualized samples.
    """
    fig, axs = subplots(nrows=n_examples, figsize=(2, n_examples * 3))

    for i in range(n_examples):
        idx = random.randint(0, len(dataset))
        img, label = dataset[idx]
        view = img.view(28, 28).numpy()
        pred_prob, pred_label = torch.max(model(img.view(1, 784)), dim=1)
        axs[i].set_title(
            f"Class: {label}\n"
            f"Prediction: {pred_label.item()}\n"
            f"Probability: {pred_prob.item():.2f}"
        )
        axs[i].imshow(view)
    return fig


def main():
    """
    Downloads the MNIST dataset, initializes a model with parameters, and visualizes sample predictions.

    This function performs the following tasks:
    1. Downloads the MNIST dataset and applies a transformation to convert images into tensors.
    2. Initializes a feed-forward neural network and loads pretrained parameters.
    3. Visualizes random sample predictions from the dataset.

    Note:
    The images are displayed using matplotlib with a resolution of 28x28 pixels.
    """
    data_path = Path("../../..") / "data"
    # Load parameters and initialize model
    weights, biases, output_weights, output_biases = load_parameters_from_files(data_path)
    mnist_model = FeedForwardNetwork(
        n_features=784,
        hidden_layer_sizes=[16, 16],
        activation_functions=[relu, relu],
        n_classes=10,
    )
    mnist_model.load_parameters(weights, biases, output_weights, output_biases)
    print(mnist_model)
    # Get MNIST dataset
    dataset = MNIST(
        str(data_path / "mnist"), train=False, transform=ToTensor(), download=True
    )
    # Visualize samples
    fig = visualize_sample_predictions(dataset, mnist_model, n_examples=2)
    pyplot.show()
    fig.savefig(Path("../../..") / "results" / "mnist.png")
    # Evaluate model
    evaluate_network(mnist_model, dataset)


if __name__ == "__main__":
    main()
