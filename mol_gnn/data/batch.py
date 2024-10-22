from typing import NamedTuple

from torch import Tensor

from mol_gnn.data.models.graph import BatchedGraph


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
