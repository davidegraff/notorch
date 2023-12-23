from abc import abstractmethod
from typing import Generic, TypeVar

from numpy.typing import NDArray

T = TypeVar("T")


class VectorFeaturizer(Generic[T]):
    """A :class:`VectorFeaturizer` calculates feature vectors of inputs."""

    @abstractmethod
    def __len__(self) -> int:
        """the length of the feature vector"""

    @abstractmethod
    def __call__(self, x: T) -> NDArray:
        """Featurize the input :attr:`x`"""
