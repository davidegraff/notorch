from copy import copy
import torch.nn as nn

from notorch.conf import DEFAULT_HIDDEN_DIM
from notorch.data.models.point_cloud import PointCloud


class PointCloudEmbed(nn.Module):
    def __init__(self, num_node_types: int, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.node = nn.EmbeddingBag(num_node_types, hidden_dim, mode="sum")

    def forward(self, P: PointCloud) -> PointCloud:
        P_emb = copy(P)
        P_emb.X = self.node(P_emb.X)

        return P_emb
