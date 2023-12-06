from torch import Tensor, nn

from mol_gnn.nn.message_passing import MessagePassing, Aggregation

GraphEncoderInput = tuple[
    Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor
]


class GraphEncoder(nn.Module):
    def __init__(self, conv: MessagePassing, agg: Aggregation):
        super().__init__()

        self.conv = conv
        self.agg = agg
    
    @property
    def output_dim(self) -> int:
        return self.conv.output_dim
    
    def forward(
        self,
        V: Tensor,
        E: Tensor,
        edge_index: Tensor,
        rev_index: Tensor | None,
        V_d: Tensor | None,
        batch: Tensor,
        n: int,
    ) -> Tensor:
        H_v = self.conv(V, E, edge_index, rev_index, V_d)
        H, _ = self.agg(H_v, batch, n)

        return H