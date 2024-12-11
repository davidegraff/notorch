from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol


class Transform[S, T, T_batched](Protocol):
    """
    A :class:`Transform` transforms an input of type ``S`` to an output of type ``T`` and knows how
    to collate the respective outputs into a batched form of type ``T_batched``.
    """
    def __call__(self, input: S) -> T: ...

    def collate(self, inputs: Collection[T]) -> T_batched: ...


@dataclass
class Pipeline[S, T, T_batched](Transform[S, T, T_batched]):
    transforms: Sequence[Transform]

    def collate(self, inputs: Collection[T]) -> T_batched:
        return self.transforms[-1].collate(inputs)

    def __call__(self, input: S) -> T:
        output = input
        for transform in self.transforms:
            output = transform(output)

        return output   # type: ignore


# @dataclass
# class JoinColumns:
#     columns: list[str]
#     out_key: str

#     def collate(self, samples: dict) -> Num[Tensor, "n t"]:
#         inputs = [sample[self.out_key] for sample in samples]

#         return {self.out_key: torch.stack(inputs)}

#     def __call__(self, sample: dict) -> Num[ArrayLike, "t"]:
#         sample[self.out_key] = [sample[column] for column in self.columns]

