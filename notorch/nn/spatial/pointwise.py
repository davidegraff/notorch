from copy import copy

import torch.nn as nn

from notorch.conf import DEFAULT_HIDDEN_DIM
from notorch.data.models.point_cloud import PointCloud


class PointwiseEmbed(nn.Module):
    def __init__(self, num_node_types: int, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.node = nn.EmbeddingBag(num_node_types, hidden_dim, mode="sum")

    def forward(self, P: PointCloud) -> PointCloud:
        P_emb = copy(P)
        P_emb.node_feats = self.node(P_emb.node_feats)

        return P_emb


class Pointwise(nn.Module):
    """Apply a module to the node features."""

    def __init__(self, module: nn.Module):
        super().__init__()

        self.module = module

    def forward(self, P: PointCloud) -> PointCloud:
        P_emb = copy(P)
        P_emb.node_feats = self.module(P_emb.node_feats)

        return P_emb
