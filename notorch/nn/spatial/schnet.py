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
        radius: float,
        d_min: float,
        d_max: float,
        num_bases: int,
        hidden_dim: int,
        act: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        rbf = RBFEmbedding(d_min, d_max, num_bases)

        self.radius = radius
        self.W = nn.Sequential(
            rbf,
            nn.Linear(rbf.num_bases, hidden_dim, bias=False),
            act(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            act(),
        )

    def forward(
        self, X: Float[Tensor, "V d_h"], R: Float[Tensor, "V d_r"], batch_index: Int[Tensor, "V"]
    ) -> Float[Tensor, "V d_h"]:
        src, dest = radius_graph(R, self.radius, batch_index).unbind(0)
        D_ij = (R[src] - R[dest]).norm(p=2, dim=-1)
        M_ij = self.W(D_ij)
        H = scatter_sum(X[src] * M_ij, dest, dim=0, dim_size=len(X))

        return H

    def extra_repr(self) -> str:
        return f"radius={self.radius:0.1f}"


class InteractionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        d_min: float,
        d_max: float,
        num_bases: int,
        radius: float,
        act: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.cfconv = ContinuousFilterConvolution(radius, d_min, d_max, num_bases, hidden_dim, act)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), act(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self, X: Float[Tensor, "V d_in"], R: Float[Tensor, "V d_r"], batch_index: Int[Tensor, "V"]
    ) -> Float[Tensor, "V d_out"]:
        H = self.W(X)
        H = self.cfconv(X, R, batch_index)
        V = self.update(H)

        return V


class SchnetBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        d_min: float,
        d_max: float,
        num_bases: int,
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
            X = layer(X, P.R, P.batch_index)

        return BatchedPointCloud(X, P.R, batch_index=P.batch_index, size=len(P), device_=P.device)
