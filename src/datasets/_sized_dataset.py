import abc
from typing import Sized, TypeVar, Generic

from torch.utils.data import Dataset

T = TypeVar("T")


class SizedDataset(Dataset[T], Generic[T], Sized, abc.ABC):
    """
    An abstract class that represents a dataset which combines the properties of
    both a generic dataset and one that has a definable size.

    The `SizedDataset` class is designed to extend the functionalities of a typical
    dataset by ensuring that it can be indexed, iterated, and has a definable size.
    This class makes use of Python's abstract base classes (ABCs) to enforce subclasses
    to provide implementations for essential functionalities.

    Inheriting from:
    - `Generic[T]`: Allows type hinting, ensuring type consistency.
    - `Dataset[T]`: Ensures the dataset can be indexed and iterated upon.
    - `Sized`: Ensures the dataset has a definable length (size).
    - `abc.ABC`: Makes the class abstract, meaning it cannot be instantiated on its own
                 and must be subclassed.

    Note:
    Subclasses that derive from this class must implement the required methods
    of the parent classes. This typically includes methods like `__getitem__`
    for indexing and `__len__` to return the dataset's size.

    Attributes:
    T: The type of elements in the dataset.
    """

    ...
