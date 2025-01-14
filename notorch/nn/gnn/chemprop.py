from __future__ import annotations

from jaxtyping import Float, Int
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter

from notorch.data.models.graph import BatchedGraph, Graph
from notorch.nn.residual import Residual
from notorch.types import Reduction


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
        self.reduce = reduce
        self.update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias), nn.Dropout(dropout))

    def forward(
        self,
        edge_feats: Float[Tensor, "E d_h"],
        node_feats: Float[Tensor, "V *"],
        edge_index: Int[Tensor, "2 E"],
        rev_index: Int[Tensor, "E"],
    ) -> Float[Tensor, "E d_h"]:
        src, dest = edge_index.unbind(0)

        edge_hiddens = self.act(edge_feats)
        messages = edge_hiddens
        node_messages = scatter(messages, dest, dim=0, dim_size=len(node_feats), reduce=self.reduce)
        edge_messages = node_messages[src] - messages[rev_index]
        edge_hiddens = self.update(edge_messages)

        return edge_hiddens

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
            layers = [Residual(layer) for layer in layers]

        self.layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim
        self.reduce = reduce

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward[T: (Graph | BatchedGraph)](self, G: T) -> T:
        src, dest = G.edge_index.unbind(0)
        edge_hiddens = G.node_feats[src] + G.edge_feats
        for layer in self.layers:
            edge_hiddens = layer(edge_hiddens, G.node_feats, G.edge_index, G.rev_index)
        node_hiddens = scatter(edge_hiddens, dest, dim=0, dim_size=G.num_nodes, reduce=self.reduce)

        return G.update(node_feats=node_hiddens, edge_feats=edge_hiddens)
