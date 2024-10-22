from torch import Tensor, nn
from mol_gnn.data.models.graph import BatchedGraph

from mol_gnn.nn.message_passing import MessagePassing
from mol_gnn.nn import Aggregation

GraphEncoderInput = tuple[BatchedGraph, Tensor | None, int]


class GraphEncoder(nn.Module):
    def __init__(self, conv: MessagePassing, agg: Aggregation):
        super().__init__()

        self.conv = conv
        self.agg = agg

    @property
    def output_dim(self) -> int:
        return self.conv.output_dim

    def forward(self, G: BatchedGraph, V_d: Tensor | None, n: int) -> Tensor:
        H_v = self.conv(G, V_d)
        H, _ = self.agg(H_v, G.batch_node_index, n)

        return H
