from collections.abc import Sequence
from typing import Protocol

from torch import nn, Tensor

from mol_gnn.data.models.graph import BatchedGraph, Graph



MessagePassingInput = tuple[BatchedGraph, Tensor | None]


class MessagePassingLayer(nn.Module, Protocol):
    """A :class:`MessagePassing` module encodes a batch of molecular graphs using message passing
    to learn vertex-level hidden representations."""

    def forward(self, G: Graph) -> Graph:
        """Encode a Graph."""


class MessagePassingBlock(MessagePassingLayer):
    layers: Sequence[MessagePassingLayer]
