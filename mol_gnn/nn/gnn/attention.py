from typing import Literal

from torch import einsum, nn
from torch_scatter import scatter, scatter_softmax

from mol_gnn.conf import DEFAULT_HIDDEN_DIM
from mol_gnn.data.models.graph import Graph

Reduction = Literal["mean", "sum", "min", "max"]


class GATv2Layer(nn.Module):
    """something resemembling GATv2 but not quite"""

    def __init__(
        self, input_dim: int = DEFAULT_HIDDEN_DIM, output_dim: int | None = None, slope: float = 0.2
    ):
        output_dim = input_dim if output_dim is None else output_dim

        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        self.update = nn.Linear(input_dim, output_dim)
        self.act = nn.LeakyReLU(slope)
        self.a = nn.Linear(output_dim, 1)

    def forward(self, G: Graph) -> Graph:
        src, dest = G.edge_index

        Q = self.W_q(G.V[src])
        K = self.W_k(G.V[dest])
        V = self.W_v(G.V[dest])
        bias = self.W_e(G.E)

        scores = self.a(self.act(Q + K + bias))
        alpha = scatter_softmax(scores, dest, dim=0)
        H = scatter(alpha * V, dest, dim=0, dim_size=len(G), reduce="sum")

        return Graph(H, G.E, G.edge_index, G.rev_index)


class MultiheadedSelfAttentionLayer(nn.Module):
    """scaled dot product attention with additive bias for edges"""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        embed_dim: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()

        embed_dim = node_dim if embed_dim is None else embed_dim
        output_dim = embed_dim if output_dim is None else output_dim

        head_dim, r = divmod(embed_dim, num_heads)
        if r != 0:
            raise ValueError

        self.W_q = nn.Sequential(
            nn.Linear(node_dim, embed_dim), nn.Unflatten(-1, (-1, num_heads, head_dim))
        )
        self.W_v = nn.Sequential(
            nn.Linear(node_dim, embed_dim), nn.Unflatten(-1, (-1, num_heads, head_dim))
        )
        self.bias = nn.Sequential(edge_dim, 1)
        self.sqrt_dk = head_dim**1 / 2

        self.W_v = nn.Sequential(
            nn.Linear(node_dim, embed_dim), nn.Unflatten(-1, (-1, num_heads, head_dim))
        )
        self.W_o = nn.Sequential(nn.Flatten(-2, -1), nn.Linear(embed_dim, output_dim))

    def forward(self, G: Graph) -> Graph:
        src, dest = G.edge_index

        Q = self.W_q(G.V[src])
        K = self.W_k(G.V[dest])
        V = self.W_v(G.V[dest])  # (E, h, d)
        bias = self.bias(G.E)

        scores = einsum("Ehd,Ehd->Eh", Q, K) / self.sqrt_dk + bias
        alpha = scatter_softmax(scores, dest, dim=0).unsqueeze(-1)
        H = scatter(alpha * V, dest, dim=0, dim_size=len(G), reduce="sum")
        O = self.W_o(H)

        return Graph(O, G.E, G.edge_index, G.rev_index)
