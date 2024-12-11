from collections.abc import Collection, Sequence
from dataclasses import dataclass
import textwrap
from typing import Protocol

from mol_gnn.conf import REPR_INDENT


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

        return output  # type: ignore

    def __repr__(self) -> str:
        text = "\n".join(f"({i}): {transform}" for i, transform in enumerate(self.transforms))

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])

