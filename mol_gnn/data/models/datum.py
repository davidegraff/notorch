from typing import TypedDict

from torch import Tensor

from mol_gnn.data.models.graph import Graph


class Datum(TypedDict):
    G: Graph
    V_d: Tensor | None
    x_f: Tensor | None
    y: Tensor | None
    weight: float
    lt_mask: Tensor | None
    gt_mask: Tensor | None
