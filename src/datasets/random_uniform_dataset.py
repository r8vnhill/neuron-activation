from typing import final

import torch

from datasets import RandomDataset


@final
@final
class RandomUniformDataset(RandomDataset):
    """
    `RandomUniformDataset` generates a synthetic dataset filled with samples drawn from a
    uniform distribution over the interval [0, 1) and corresponding random labels.

    This specialized dataset class derives from `RandomDataset`.
    The data samples (features) in `RandomUniformDataset` are generated based on a uniform
    distribution over [0, 1), making it suitable for scenarios where uniformly distributed
    data is preferable or required for testing or simulation.

    :param n_examples: Total number of samples in the dataset.
    :param n_features: Dimensionality of each data sample.
    :param n_classes: Total distinct class labels.

    Example:
    --------
    To generate a dataset with 100 samples, each with 5 features drawn from a uniform distribution
    and belonging to one of 3 classes, you'd do:

    >>> dataset = RandomUniformDataset(100, 5, 3)

    Note:
    -----
    The label-generation mechanism in `RandomUniformDataset` derives from its parent class,
    `RandomDataset`.
    Labels are integers chosen randomly from the range [0, n_classes).
    """

    def __init__(self, n_examples: int, n_features: int, n_classes: int):
        super().__init__(n_examples, n_features, n_classes,
                         # Generates data based on uniform distribution over [0, 1)
                         lambda x, y: torch.rand(x, y))
