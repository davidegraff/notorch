from collections.abc import Collection, Sequence
from dataclasses import dataclass
import textwrap
from typing import ClassVar, Protocol

from notorch.conf import REPR_INDENT
from notorch.data.models.graph import BatchedGraph, Graph


class Transform[S, T, T_batched](Protocol):
    """
    A :class:`Transform` transforms an input of type ``S`` to an output of type ``T`` and knows how
    to collate the respective outputs into a batched form of type ``T_batched``.
    """

    _in_key_: ClassVar[str]
    _out_key_: ClassVar[str]

    def __call__(self, input: S) -> T: ...
    def collate(self, inputs: Collection[T]) -> T_batched: ...


class GraphTransform[S](Transform[S, Graph, BatchedGraph]):
    num_node_types: int
    num_edge_types: int


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
