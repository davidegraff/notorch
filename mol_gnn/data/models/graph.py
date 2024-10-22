from dataclasses import InitVar, dataclass
from typing import Iterable, Self

import torch
from torch import Tensor


@dataclass(repr=False, eq=False)
class Graph:
    """A :class:`Graph` represents the feature representation of graph."""

    V: Tensor
    """a tensor of shape ``|V| x d_v`` containing the vertex features of the graph"""
    E: Tensor
    """a tensor of shape ``|E| x d_e`` containing the edge features of the graph"""
    edge_index: Tensor
    """a tensor of shape ``2 x |E|`` containing the edges of the graph in COO format"""
    rev_index: Tensor
    """a tensor of shape ``|E|`` that maps from an edge index to the index of the source of the
    reverse edge in :attr:`edge_index` attribute."""

    @property
    def A(self) -> Tensor:
        num_nodes = self.V.shape[0]
        src, dest = self.edge_index.unbind(0)

        A = torch.zeros(num_nodes, num_nodes)
        A[src, dest] = 1

        return A

@dataclass(repr=False, eq=False)
class BatchedGraph(Graph):
    """A :class:`BatchedMolGraph` represents a batch of individual :class:`Graph`s."""

    batch_node_index: Tensor
    """the index of the parent :class:`Graph` in the batched graph"""
    batch_edge_index: Tensor
    """the index of the parent :class:`Graph` in the batched graph"""
    size: InitVar[int] | None = None

    def __post_init__(self, size: int | None):
        self.__size = self.batch_node_index.max() + 1 if size is None else size

    @classmethod
    def from_graphs(cls, Gs: Iterable[Graph]):
        Vs = []
        Es = []
        batch_edge_indices = []
        rev_indices = []
        batch_node_indices = []
        batch_edge_indices = []
        offset = 0

        for i, G in enumerate(Gs):
            Vs.append(G.V)
            Es.append(G.E)
            batch_edge_indices.append(G.edge_index + offset)
            rev_indices.append(G.rev_index + offset)
            batch_node_indices.append([i] * len(G.V))
            batch_edge_indices.append([i] * len(G.E))

            offset += len(G.V)

        V = torch.cat(Vs).float()
        E = torch.cat(Es).float()
        edge_index = torch.cat(batch_edge_indices, dim=1).long()
        rev_index = torch.cat(rev_indices).long()
        batch_node_index = torch.cat(batch_node_indices).long()
        batch_edge_index = torch.cat(batch_edge_indices).long()
        size = i + 1

        return cls(V, E, edge_index, rev_index, batch_node_index, batch_edge_index, size)

    # def to_graphs(self) -> list[Graph]:
    #     split_sizes = self.batch_node_index.bincount(minlength=len(self))

    #     Vs = self.V.split_with_sizes(split_sizes)

    def __len__(self) -> int:
        """the number of individual :class:`Graph`s in this batch"""
        return self.__size

    def to(self, device: str | torch.device) -> Self:
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_index = self.rev_index.to(device)
        self.batch_node_index = self.batch_node_index.to(device)
        self.batch_edge_index = self.batch_edge_index.to(device)

        return self
