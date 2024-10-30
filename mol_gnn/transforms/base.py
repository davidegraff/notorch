from abc import abstractmethod
from collections.abc import Iterable, Sequence, Sized
from dataclasses import dataclass
from typing import overload

from jaxtyping import Num
from torch import Tensor


class Transform[S, T]:
    @overload
    def __call__(self, input: S) -> T: ...

    @overload
    def __call__(self, input: Iterable[S]) -> Sequence[T]: ...

    @abstractmethod
    def __call__(self, input: S | Iterable[S]) -> T | Sequence[T]:
        pass


class TensorTransform[S](Sized, Transform[S, Num[Tensor, "*n d"]]):
    """A :class:`VectorFeaturizer` transforms inputs into tensors."""

    @abstractmethod
    def __call__(self, input: S | Iterable[S]) -> Num[Tensor, "*n d"]:
        pass

@dataclass
class MultiTransform[S](Transform[S]):
    transforms: list[Transform[S]]

    def __call__(self, input):
        return [transform(input) for transform in self.transforms]



class Pipeline[S, T](Transform[S, T]):
    def __call__(self, input):
        return super().__call__(input)

    # @abstractmethod
    # def __call__(self, input):
    #     return super().__call__(input)
    # def __call__(self, input):
    #     return super().__call__(input)
