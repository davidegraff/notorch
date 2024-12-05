from __future__ import annotations

from copy import copy
from typing import Literal

from torch import nn

from mol_gnn.data.models.graph import Graph

Reduction = Literal["mean", "sum", "min", "max"]


class GraphEmbedding(nn.Module):
    def __init__(self, num_node_types: int, num_edge_types: int, hidden_dim: int = 256):
        super().__init__()

        self.node = nn.EmbeddingBag(num_node_types, hidden_dim, mode="sum")
        self.edge = nn.EmbeddingBag(num_edge_types, hidden_dim, mode="sum")

    def forward(self, G: Graph) -> Graph:
        G = copy(G)
        G.V = self.embed["node"](G.V)
        G.E = self.embed["edge"](G.E)

        return G

    @property
    def num_node_types(self) -> int:
        return self.node.num_embeddings

    @property
    def num_edge_types(self) -> int:
        return self.edge.num_embeddings
