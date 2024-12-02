from typing import Annotated

from jaxtyping import Float
from torch import Tensor, nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax

from mol_gnn.conf import DEFAULT_HIDDEN_DIM
from mol_gnn.data.models.graph import BatchedGraph
from mol_gnn.nn.gnn.base import Aggregation
from mol_gnn.utils.registry import ClassRegistry


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("sum")
class Sum(Aggregation):
    def forward(self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"]) -> Float[Tensor, "b d_v"]:
        H = scatter_sum(G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("mean")
class Mean(Aggregation):
    def forward(self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"]) -> Float[Tensor, "b d_v"]:
        H = scatter_mean(G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("max")
class Max(Aggregation):
    def forward(self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"]) -> Float[Tensor, "b d_v"]:
        H, _ = scatter_max(G.V, G.batch_node_index, dim=0, dim_size=len(G))

        return H


@AggregationRegistry.register("gated")
class Gated(Aggregation):
    """A learnable aggregation gate"""

    def __init__(self, input_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()

        self.a = nn.Linear(input_dim, 1)

    def forward(self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"]) -> Float[Tensor, "b d_v"]:
        scores = self.a(G.V)
        alpha = scatter_softmax(scores, G.edge_index[1], dim=0)
        H = scatter_sum(alpha * G.V, G.batch_node_index, dim=0, dim_size=len(G)), alpha

        return H
