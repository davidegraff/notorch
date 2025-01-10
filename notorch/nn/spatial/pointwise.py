from copy import copy

import torch.nn as nn

from notorch.conf import DEFAULT_HIDDEN_DIM
from notorch.data.models.point_cloud import PointCloud
from notorch.nn.mlp import MLP


class PointwiseEmbed(nn.Module):
    def __init__(self, num_node_types: int, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.node = nn.EmbeddingBag(num_node_types, hidden_dim, mode="sum")

    def forward(self, P: PointCloud) -> PointCloud:
        P_emb = copy(P)
        P_emb.X = self.node(P_emb.X)

        return P_emb


class PointwiseMLP(nn.Module):
    """Apply an MLP to each point.

    Parameters
    ----------
    *args, **kwargs
        positional and keyword arguments to supply to :func:`notorch.nn.mlp.MLP`
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.mlp = MLP(*args, **kwargs)

    def forward(self, P: PointCloud) -> PointCloud:
        P_emb = copy(P)
        P_emb.X = self.mlp(P_emb.X)

        return P_emb
