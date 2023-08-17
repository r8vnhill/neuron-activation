from typing import final, Final

import torch

from ._sized_dataset import SizedDataset


@final
class RandomDataset(SizedDataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    A dataset class that generates random data and labels.
    This dataset is meant for simulation purposes and is not derived from real-world data.

    The RandomDataset provides random tensors for both features and labels, making it
    useful for generating dummy data for testing or demonstration purposes without
    relying on external data sources.

    :param n_examples: The number of examples in the dataset.
    :param n_features: The number of features for each example.
    :param n_classes: The number of distinct classes for labeling.

    :raises AssertionError: If any of the input parameters is less than or equal to zero.

    Note
    ----
    This class is marked as `final`, meaning it cannot be subclassed.
    """

    __n_examples: Final[int]
    __data: Final[torch.Tensor]
    __labels: Final[torch.Tensor]

    def __init__(self, n_examples: int, n_features: int, n_classes: int):
        assert n_examples > 0, "Number of examples should be greater than 0."
        assert n_features > 0, "Number of features should be greater than 0."
        assert n_classes > 0, "Number of classes should be greater than 0."

        super(RandomDataset, self).__init__()
        self.__n_examples = n_examples
        self.__data = torch.randn(n_examples, n_features)
        self.__labels = torch.randint(0, n_classes, (n_examples,))

    def __len__(self) -> int:
        """Returns the size of the dataset. """
        return self.__n_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves the data and label pair at a specified index.  """
        return self.__data[index], self.__labels[index]

