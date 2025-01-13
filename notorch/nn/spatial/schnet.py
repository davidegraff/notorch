from collections.abc import Iterable

from jaxtyping import Float, Int
from torch import Tensor
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter_sum

from notorch.data.models.point_cloud import BatchedPointCloud
from notorch.nn.rbf import RBFEmbedding
from notorch.nn.residual import Residual


class ContinuousFilterConvolution(nn.Module):
    def __init__(
        self,
        d_min: float = 0,
        d_max: float = 4.5,
        num_bases: int = 16,
        hidden_dim: int = 128,
        radius: float = 5.0,
        act: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.radius = radius
        self.W = nn.Sequential(
            RBFEmbedding(d_min, d_max, num_bases),
            nn.Linear(num_bases, hidden_dim, bias=False),
            act(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            act(),
        )

    def forward(
        self,
        node_feats: Float[Tensor, "V d_h"],
        coords: Float[Tensor, "V d_r"],
        batch_index: Int[Tensor, "V"],
    ) -> Float[Tensor, "V d_h"]:
        src, dest = radius_graph(coords, self.radius, batch_index).unbind(0)
        D_ij = (coords[src] - coords[dest]).norm(p=2, dim=-1)
        M_ij = self.W(D_ij)
        H = scatter_sum(node_feats[src] * M_ij, dest, dim=0, dim_size=len(node_feats))

        return H

    def extra_repr(self) -> str:
        return f"radius={self.radius:0.1f}"


class InteractionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        d_min: float = 0,
        d_max: float = 4.5,
        num_bases: int = 16,
        radius: float = 5.0,
        act: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.cfconv = ContinuousFilterConvolution(radius, d_min, d_max, num_bases, hidden_dim, act)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        node_feats: Float[Tensor, "V d_h"],
        coords: Float[Tensor, "V d_r"],
        batch_index: Int[Tensor, "V"],
    ) -> Float[Tensor, "V d_h"]:
        node_hiddens = self.W(node_feats)
        node_hiddens = self.cfconv(node_hiddens, coords, batch_index)
        node_hiddens = self.update(node_hiddens)

        return node_hiddens


class SchnetBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        d_min: float = 0,
        d_max: float = 4.5,
        num_bases: int = 16,
        radii: Iterable[float] = (5.0, 5.0, 5.0),
        act: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        layers = [
            InteractionLayer(hidden_dim, d_min, d_max, num_bases, radius, act) for radius in radii
        ]
        layers = [Residual(layer) for layer in layers]

        self.layers = nn.ModuleList(layers)

    def forward(self, P: BatchedPointCloud) -> BatchedPointCloud:
        for layer in self.layers:
            node_feats = layer(node_feats, P.coords, P.batch_index)

        return BatchedPointCloud(
            node_feats, P.coords, batch_index=P.batch_index, size=len(P), device_=P.device
        )
