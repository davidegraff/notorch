from math import sqrt
from typing import Annotated

from jaxtyping import Float
from torch import Tensor, nn
import torch
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax

from mol_gnn.conf import DEFAULT_HIDDEN_DIM
from mol_gnn.data.models.graph import BatchedGraph
from mol_gnn.nn.gnn.base import Aggregation
from mol_gnn.utils.registry import ClassRegistry


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("sum")
class Sum(Aggregation):
    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        H = scatter_sum(G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("mean")
class Mean(Aggregation):
    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        H = scatter_mean(G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("max")
class Max(Aggregation):
    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        H, _ = scatter_max(G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("gated")
class Gated(Aggregation):
    def __init__(self, input_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.a = nn.Linear(input_dim, 1)

    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        scores = self.a(G.V)
        alpha = scatter_softmax(scores, G.batch_node_index, dim=0, dim_size=len(G)).unsqueeze(1)
        H = scatter_sum(alpha * G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("sdp")
class SDPAttention(Aggregation):
    def __init__(self, key_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.sqrt_key_dim = sqrt(key_dim)

    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], Q: Float[Tensor, "b d_v"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        scores = torch.einsum("V d_v, V d_v -> V", Q[G.batch_node_index], G.V) / self.sqrt_key_dim
        alpha = scatter_softmax(scores, G.batch_node_index, dim=0, dim_size=len(G)).unsqueeze(1)
        H = scatter_sum(alpha * G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H
