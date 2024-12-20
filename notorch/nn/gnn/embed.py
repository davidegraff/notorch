from __future__ import annotations

from copy import copy
from typing import Literal

import torch.nn as nn

from notorch.conf import DEFAULT_HIDDEN_DIM
from notorch.data.models.graph import Graph

Reduction = Literal["mean", "sum", "min", "max"]


class GraphEmbedding(nn.Module):
    def __init__(
        self, num_node_types: int, num_edge_types: int, hidden_dim: int = DEFAULT_HIDDEN_DIM
    ):
        super().__init__()

        self.node = nn.EmbeddingBag(num_node_types, hidden_dim, mode="sum")
        self.edge = nn.EmbeddingBag(num_edge_types, hidden_dim, mode="sum")

    def forward(self, G: Graph) -> Graph:
        G_emb = copy(G)
        G_emb.V = self.node(G_emb.V)
        G_emb.E = self.edge(G_emb.E)

        return G_emb

    @property
    def num_node_types(self) -> int:
        return self.node.num_embeddings

    @property
    def num_edge_types(self) -> int:
        return self.edge.num_embeddings
