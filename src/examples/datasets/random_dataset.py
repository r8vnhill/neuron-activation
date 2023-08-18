import logging

from torch.utils.data import DataLoader

from datasets import RandomUniformDataset, RandomNormalDataset, SizedDataset, \
    BernoulliDataset


def _load_dataset(dataset: SizedDataset):
    """
    Loads the provided dataset and prints batches of data samples and corresponding
    labels.

    :param dataset: An instance of `SizedDataset` containing data samples and labels.
    """
    data = DataLoader(dataset, batch_size=5)
    for x, y in data:
        print(f"Data::{x.dtype} {x.shape}:")
        print(x)
        print(f"Labels::{y.dtype} {y.shape}:")
        print(y)


def _generate_and_load_random_uniform_data():
    """
    Generates a random uniform dataset and loads its data, printing batches of samples and
    labels.
    """
    _load_dataset(RandomUniformDataset(10, 8, 10))


def _generate_and_load_random_normal_data():
    """
    Generates a random normal dataset and loads its data, printing batches of samples and
    labels.
    """
    _load_dataset(RandomNormalDataset(10, 8, 10))


def _generate_and_load_bernoulli_data():
    """
    Generates a Bernoulli dataset and loads its data, printing batches of samples and
    labels.
    """
    _load_dataset(BernoulliDataset(10, 8, 10))


def _trim_indent(s: str) -> str:
    """
    Removes leading and trailing white spaces from each line in a multi-line string.

    :param s: Multi-line string to be processed.
    :return: A new string with indents removed from each line.
    """
    return "\n".join([line.strip() for line in s.split("\n")])


if __name__ == "__main__":
    # Displaying dataset details for Random Uniform Dataset.
    print(_trim_indent("""
    ============================================================================
    =                          Random Uniform Dataset                          =
    ============================================================================
    """))
    _generate_and_load_random_uniform_data()

    # Displaying dataset details for Random Normal Dataset.
    print(_trim_indent("""
    ============================================================================
    =                          Random Normal Dataset                           =
    ============================================================================
    """))
    _generate_and_load_random_normal_data()

    # Displaying dataset details for Bernoulli Dataset.
    print(_trim_indent("""
    ============================================================================
    =                             Bernoulli Dataset                            =
    ============================================================================
    """))
    _generate_and_load_bernoulli_data()
