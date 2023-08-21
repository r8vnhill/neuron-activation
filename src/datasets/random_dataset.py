import abc
from typing import Final, Callable

import torch

from ._sized_dataset import SizedDataset


class RandomDataset(SizedDataset[tuple[torch.Tensor, torch.Tensor]], abc.ABC):
    """
    `RandomDataset` facilitates the creation of a synthetic dataset filled with random
    data and corresponding labels.
    It serves as a tool primarily designed for simulation, prototyping, and testing where
    access to real-world data might be restrictive or unnecessary.

    This class generates random tensors for both data (features) and labels based on
    user-defined dimensions and classes.
    It proves useful when one needs to quickly generate dummy data for tasks like model
    testing or demonstration without relying on external datasets.

    :param n_examples: Total number of samples in the dataset.
    :param n_features: Dimensionality of each data sample.
    :param n_classes: Total distinct class labels.
    :param random_generator: A function to produce the random data tensor.
                             Allows users to define their custom random generation logic.

    :raises AssertionError: If any of the parameters `n_examples`, `n_features`, or
                            `n_classes` is less than or equal to zero.
    """

    __n_examples: Final[int]

    def __init__(
        self,
        n_examples: int,
        n_features: int,
        n_classes: int,
        random_generator: Callable[[int, int], torch.Tensor],
    ):
        super().__init__()

        # Assert input conditions
        assert n_examples > 0, "Number of examples should be greater than 0."
        assert n_features > 0, "Number of features should be greater than 0."
        assert n_classes > 0, "Number of classes should be greater than 0."

        self.__n_examples = n_examples
        self._data = random_generator(n_examples, n_features)
        self._labels = torch.randint(0, n_classes, (n_examples,))

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.__n_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches a specific data-label pair based on the provided index.

        :param index: Index of the desired data-label pair.
        :return: A tuple containing the data sample and its corresponding label.
        """
        return self._data[index], self._labels[index]
