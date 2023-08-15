import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from devices import Device
from network import NeuralNetwork


def evaluate_network(
    network: NeuralNetwork,
    dataset: VisionDataset,
    batch_size: int = 100,
    device: Device = Device.CPU,
):
    """
    Evaluates the performance of a neural network on a given vision dataset.

    This function iterates over the dataset using batches, computes predictions for each
    batch using the provided network, and tracks the number of correct predictions.
    At the end, it prints the accuracy of the network on the dataset.

    :param network: The neural network model to be evaluated.
    :param dataset: The dataset on which the network is evaluated.
    :param batch_size: The size of the batches in which the dataset is divided for
                       evaluation.
                       Default is 100.
    :param device: The device on which the computations are performed (CPU or GPU).

    __Note:__

    - This function assumes that the network's forward method outputs raw scores (logits)
      for each class.
    - The accuracy is computed as the percentage of correct predictions over the total
      number of samples in the dataset.
    """
    network.to(device)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    n_correct = 0
    for i, (x, y) in enumerate(data_loader):
        view: torch.Tensor = x.view(-1, network.input_size).to(device)
        predictions: torch.Tensor = torch.max(network(view), dim=1)[1]
        n_correct += torch.sum(torch.eq(predictions, y.to(device))).item()

        if i % 10 == 0:
            print(f"{i * batch_size}/{len(dataset)}")

    print(f"Accuracy: {(n_correct / len(dataset) * 100):.2f}%")
