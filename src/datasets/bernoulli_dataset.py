from typing import final

import torch

from .random_dataset import RandomDataset

__all__ = ["BernoulliDataset", "RandomDataset"]


@final
class BernoulliDataset(RandomDataset):
    """
    `BernoulliDataset` creates a synthetic dataset filled with Bernoulli-distributed
    random data and corresponding labels.

    This specialized dataset class derives from `RandomDataset`.
    The data samples (features) in `BernoulliDataset` are generated based on the Bernoulli
    distribution, wherein each entry is a random binary value (0 or 1), with the
    probability determined by the outcome of a Bernoulli trial (i.e., a random draw from a
    uniform distribution, followed by thresholding at 0.5).

    It can serve various purposes, such as simulation, prototyping, and testing in
    situations that require binary or two-state data points.

    :param n_examples: Total number of samples in the dataset.
    :param n_features: Dimensionality of each data sample.
    :param n_classes: Total distinct class labels.

    Example:
    -------
    If you want to generate a dataset with 100 samples, each having 5 binary features and
    belonging to one of 3 classes, you'd do:

        >>> dataset = BernoulliDataset(100, 5, 3)

    Note
    ----
    The `BernoulliDataset` derives its label-generation mechanism from its parent class,
    `RandomDataset`.
    Labels are integers randomly chosen from the range [0, n_classes).
    """

    def __init__(self, n_examples: int, n_features: int, n_classes: int):
        super().__init__(
            n_examples,
            n_features,
            n_classes,
            lambda x, y: torch.bernoulli(torch.rand(x, y))
            # Generates Bernoulli-distributed random data
        )
