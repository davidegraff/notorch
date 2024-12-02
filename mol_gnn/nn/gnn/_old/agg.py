from abc import abstractmethod

from torch import Tensor, nn

from mol_gnn.nn.agg import Aggregation


class MessageAggregation(nn.Module):
    def __init__(self, agg: Aggregation):
        super().__init__()

        self.agg = agg

    @property
    def mult(self) -> int:
        """the multiplier to apply to the output dimension"""
        return 1

    @abstractmethod
    def forward(
        self, M: Tensor, edge_index: Tensor, dim_size: int | None, rev_index: Tensor
    ) -> Tensor:
        """Aggregate the incoming messages"""


class NodeAggregation(MessageAggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, dim_size: int | None, rev_index: Tensor
    ) -> Tensor:
        _, dest = edge_index

        return self.agg(M, dest, dim_size, rev_index=rev_index)[0]


class EdgeAggregation(MessageAggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, dim_size: int | None, rev_index: Tensor
    ) -> Tensor:
        src, dest = edge_index

        M = (M + M[rev_index]) / 2
        M_v, _ = self.agg(M, dest, dim_size, rev_index=rev_index)

        return M_v[src]


class DirectedEdgeAggregation(MessageAggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, dim_size: int | None, rev_index: Tensor
    ) -> Tensor:
        src, dest = edge_index

        M_v, w = self.agg(M, dest, dim_size, rev_index=rev_index)

        return M_v[src] - w * M[rev_index]
