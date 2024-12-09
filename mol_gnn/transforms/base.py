from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class Transform[S, T, T_batched](Protocol):
    def __call__(self, input: T) -> S: ...

    def collate(self, inputs: list[T]) -> T_batched: ...


@dataclass
class Pipeline[S, T, T_batched](Transform[S, T, T_batched]):
    transforms: Sequence[Transform]

    def collate(self, inputs: list[T]) -> T_batched:
        return self.transforms[-1].collate(inputs)

    def __call__(self, input: S) -> T:
        output = input
        for transform in self.transforms:
            output = transform(output)

        return output


@dataclass
class ManagedTransform[S, T, T_batched](Transform[S, T, T_batched]):
    transform: Transform[S, T, T_batched]
    in_key: str
    out_key: str

    def collate(self, inputs: list[T]) -> T_batched:
        return self.transform.collate(inputs)

    def __call__(self, sample: dict) -> dict:
        sample[self.out_key] = self.transform(sample[self.in_key])

        return sample
