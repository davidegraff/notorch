from __future__ import annotations

import torch.nn as nn

from notorch.conf import DEFAULT_HIDDEN_DIM
from notorch.data.models.graph import Graph
from notorch.transforms.base import GraphTransform
from notorch.transforms.conf import DEFAULT_NUM_ATOM_TYPES, DEFAULT_NUM_BOND_TYPES


class GraphEmbedding(nn.Module):
    def __init__(
        self,
        num_node_types: int = DEFAULT_NUM_ATOM_TYPES,
        num_edge_types: int = DEFAULT_NUM_BOND_TYPES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()

        self.node = nn.EmbeddingBag(num_node_types, hidden_dim, mode="sum")
        self.edge = nn.EmbeddingBag(num_edge_types, hidden_dim, mode="sum")

    def forward(self, G: Graph) -> Graph:
        return G.update(node_feats=self.node(G.node_feats), edge_feats=self.edge(G.edge_feats))

    @property
    def num_node_types(self) -> int:
        return self.node.num_embeddings

    @property
    def num_edge_types(self) -> int:
        return self.edge.num_embeddings

    @classmethod
    def from_transform(cls, transform: GraphTransform, **kwargs) -> GraphEmbedding:
        return cls(transform.num_node_types, transform.num_edge_types, **kwargs)
