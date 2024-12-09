from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import overload


class Transform[S, T]:
    @overload
    def __call__(self, input: S) -> T: ...

    @overload
    def __call__(self, input: Iterable[S]) -> Sequence[T]: ...

    @abstractmethod
    def __call__(self, input: S | Iterable[S]) -> T | Sequence[T]:
        pass


@dataclass
class Pipeline[S, T](Transform):
    transforms: Sequence[Transform]

    def __call__(self, input: S) -> T:
        output = input
        for transform in self.transforms:
            output = transform(output)

        return output

    # @abstractmethod
    # def __call__(self, input):
    #     return super().__call__(input)
    # def __call__(self, input):
    #     return super().__call__(input)
