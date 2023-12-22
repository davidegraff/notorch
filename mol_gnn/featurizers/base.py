from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class Featurizer(Callable, Generic[T]):
    """A :class:`Featurizer` calculates feature vectors of RDKit molecules."""

    @abstractmethod
    def __len__(self) -> int:
        """the length of the feature vector"""

    @abstractmethod
    def __call__(self, x: T) -> np.ndarray:
        """Featurize the input :attr:`x`"""
