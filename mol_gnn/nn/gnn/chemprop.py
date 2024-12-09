from __future__ import annotations

from copy import copy
from typing import Literal

import torch.nn as nn
from torch_scatter import scatter

from mol_gnn.data.models.graph import Graph
from mol_gnn.nn.residual import Residual

Reduction = Literal["mean", "sum", "min", "max"]


class ChempropLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        act: type[nn.Module] = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        reduce: Reduction = "sum",
    ):
        super().__init__()

        self.act = act()
        self.message = nn.Identity()
        self.reduce = reduce
        self.update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias), nn.Dropout(dropout))

    def forward(self, G: Graph) -> Graph:
        src, dest = G.edge_index.unbind(0)

        H_uv = self.act(G.E)
        M_uv = H_uv
        M_v = scatter(M_uv, dest, dim=0, dim_size=G.num_nodes, reduce=self.reduce)
        M_uv = M_v[src] - M_uv[G.rev_index]
        H_uv = self.update(M_uv)

        return Graph(G.V, H_uv, G.edge_index, G.rev_index)

    def extra_repr(self):
        return f"(reduce): {self.reduce}"


class ChempropBlock(nn.Module):
    layers: list[Residual[ChempropLayer]]

    def __init__(
        self,
        hidden_dim: int = 256,
        act: type[nn.Module] = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
        depth: int = 3,
        residual: bool = True,
        shared: bool = False,
        reduce: Reduction = "sum",
    ):
        super().__init__()

        if shared:
            layers = [ChempropLayer(hidden_dim, act, bias, dropout, reduce)] * depth
        else:
            layers = [ChempropLayer(hidden_dim, act, bias, dropout, reduce) for _ in range(depth)]

        if residual:
            # residual connection on edge features
            layers = [
                Residual(
                    layer, lambda g1, g2: Graph(g1.V, g1.E + g2.E, g1.edge_index, g1.rev_index)
                )
                for layer in layers
            ]

        self.block = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
        self.reduce = reduce
        self.depth = len(self.block)

    def forward(self, G: Graph) -> Graph:
        G_t = copy(G)
        G_t.E = self.embed["node"](G.V) + self.embed["edge"](G.E)
        G_t = self.block(G_t)
        G_t.V = scatter(
            G_t.E, G_t.edge_index[0], dim=0, dim_size=G.num_nodes, rev_index=G.rev_index
        )

        return G_t
