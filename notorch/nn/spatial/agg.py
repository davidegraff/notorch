from math import sqrt

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum

from notorch.conf import DEFAULT_HIDDEN_DIM
from notorch.data.models.point_cloud import BatchedPointCloud


class Sum(nn.Module):
    def forward(self, P: BatchedPointCloud) -> Float[Tensor, "b d_v"]:
        return scatter_sum(P.node_feats, P.batch_index, dim=0, dim_size=len(P))


class Mean(nn.Module):
    def forward(self, P: BatchedPointCloud) -> Float[Tensor, "b d_v"]:
        return scatter_mean(P.node_feats, P.batch_index, dim=0, dim_size=len(P))


class Max(nn.Module):
    def forward(self, P: BatchedPointCloud) -> Float[Tensor, "b d_v"]:
        H, _ = scatter_max(P.node_feats, P.batch_index, dim=0, dim_size=len(P))

        return H


class Gated(nn.Module):
    def __init__(self, input_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.a = nn.Linear(input_dim, 1)

    def forward(self, P: BatchedPointCloud) -> Float[Tensor, "b d_v"]:
        scores = self.a(P.node_feats)
        alpha = scatter_softmax(scores, P.node_feats, dim=0, dim_size=len(P.node_feats)).unsqueeze(
            1
        )
        H = scatter_sum(alpha * P.node_feats, P.batch_index, dim=0, dim_size=len(P.node_feats))

        return H


class SDPAttention(nn.Module):
    def __init__(self, key_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.sqrt_key_dim = sqrt(key_dim)

    def forward(
        self, P: BatchedPointCloud, *, Q: Float[Tensor, "b d_v"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        scores = (
            torch.einsum("V d_v, V d_v -> V", Q[P.batch_index], P.node_feats) / self.sqrt_key_dim
        )
        alpha = scatter_softmax(scores, P.batch_index, dim=0, dim_size=len(P.node_feats)).unsqueeze(
            1
        )
        H = scatter_sum(alpha * P.node_feats, P.batch_index, dim=0, dim_size=len(P.node_feats))

        return H
