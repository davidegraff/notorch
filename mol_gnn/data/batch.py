from dataclasses import dataclass, field, InitVar
from functools import cache
from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch import Tensor

from mol_gnn.data.models.graph import Graph


@dataclass(repr=False, eq=False)
class BatchedGraph:
    """A :class:`BatchedMolGraph` represents a batch of individual :class:`Graph`s.

    It has all the attributes of a :class:`Graph` with the addition of the :attr:`batch`
    attribute. This class is intended for use with data loading, so it uses :obj:`~torch.Tensor`s
    to store data
    """

    Gs: InitVar[Iterable[Graph]]
    """A list of individual :class:`Graph`s to be batched together"""
    V: Tensor = field(init=False)
    """the atom feature matrix"""
    E: Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_index: Tensor | None = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in :attr:`edge_index`."""
    batch: Tensor = field(init=False)
    """the index of the parent :class:`Graph` in the batched graph"""

    def __post_init__(self, Gs: Iterable[Graph]):
        Vs = []
        Es = []
        edge_indexes = []
        rev_indexes = []
        batch_indexes = []
        offset = 0

        for i, G in enumerate(Gs):
            Vs.append(G.V)
            Es.append(G.E)
            edge_indexes.append(G.edge_index + offset)
            rev_indexes.append(G.rev_index + offset)
            batch_indexes.append([i] * len(G.V))

            offset += len(G.V)

        self.V = torch.from_numpy(np.concatenate(Vs)).float()
        self.E = torch.from_numpy(np.concatenate(Es)).float()
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long()
        self.rev_index = torch.from_numpy(np.concatenate(rev_indexes)).long()
        self.batch = torch.from_numpy(np.concatenate(batch_indexes)).long()

    @cache
    def __len__(self) -> int:
        """the number of individual :class:`Graph`s in this batch"""
        return self.batch.max() + 1

    def to(self, device: str | torch.device):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_index = self.rev_index.to(device)
        self.batch = self.batch.to(device)


class MpnnBatch(NamedTuple):
    G: BatchedGraph
    V_d: Tensor | None
    X_f: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


class MultiInputMpnnBatch(NamedTuple):
    Gs: list[BatchedGraph]
    V_ds: list[Tensor]
    X_f: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None
