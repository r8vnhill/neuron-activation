from typing import final

import torch

from datasets import RandomDataset


@final
class RandomNormalDataset(RandomDataset):
    """
    `RandomNormalDataset` generates a synthetic dataset filled with samples drawn from a
    standard normal distribution (mean=0, std=1) and corresponding random labels.

    This specialized dataset class derives from `RandomDataset`.
    The data samples (features) in `RandomNormalDataset` are generated based on a standard
    normal distribution, making it ideal for scenarios where normally distributed data
    is preferable or required for testing or simulation.

    :param n_examples: Total number of samples in the dataset.
    :param n_features: Dimensionality of each data sample.
    :param n_classes: Total distinct class labels.

    Example:
    --------
    To generate a dataset with 100 samples, each with 5 features drawn from a standard
    normal distribution and belonging to one of 3 classes, you'd do:

    >>> dataset = RandomNormalDataset(100, 5, 3)

    Note:
    -----
    The label-generation mechanism in `RandomNormalDataset` derives from its parent class,
    `RandomDataset`.
    Labels are integers chosen randomly from the range [0, n_classes).
    """

    def __init__(self, n_examples: int, n_features: int, n_classes: int):
        super().__init__(n_examples, n_features, n_classes,
                         # Generates data based on standard normal distribution
                         lambda x, y: torch.randn(x, y))

