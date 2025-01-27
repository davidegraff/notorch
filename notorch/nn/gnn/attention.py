from jaxtyping import Float, Int
from torch import Tensor, einsum, nn
from torch_scatter import scatter, scatter_softmax

from notorch.conf import DEFAULT_HIDDEN_DIM


class GATv2Layer(nn.Module):
    """something resemembling GATv2 but not quite"""

    def __init__(
        self, input_dim: int = DEFAULT_HIDDEN_DIM, output_dim: int | None = None, slope: float = 0.2
    ):
        output_dim = input_dim if output_dim is None else output_dim

        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        self.act = nn.LeakyReLU(slope)
        self.a = nn.Linear(output_dim, 1)

    def forward(
        self,
        node_feats: Float[Tensor, "V d_v"],
        edge_feats: Float[Tensor, "E d_e"],
        edge_index: Int[Tensor, "2 E"],
    ) -> Float[Tensor, "V d_o"]:
        src, dest = edge_index.unbind(0)

        Q = self.W_q(node_feats[src])
        K = self.W_k(node_feats[dest])
        V = self.W_v(node_feats[dest])
        bias = self.W_e(edge_feats)

        scores = self.a(self.act(Q + K + bias))
        alpha = scatter_softmax(scores, dest, dim=0)
        node_hiddens = scatter(alpha * V, dest, dim=0, dim_size=len(node_feats), reduce="sum")

        return node_hiddens


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
        output_dim = node_dim if output_dim is None else output_dim

        head_dim, r = divmod(embed_dim, num_heads)
        if r != 0:
            raise ValueError

        self.W_q = nn.Sequential(
            nn.Linear(node_dim, embed_dim), nn.Unflatten(-1, (-1, num_heads, head_dim))
        )
        self.W_v = nn.Sequential(
            nn.Linear(node_dim, embed_dim), nn.Unflatten(-1, (-1, num_heads, head_dim))
        )
        self.bias = nn.Linear(edge_dim, 1)
        self.sqrt_dk = head_dim**1 / 2

        self.W_v = nn.Sequential(
            nn.Linear(node_dim, embed_dim), nn.Unflatten(-1, (-1, num_heads, head_dim))
        )
        self.W_o = nn.Sequential(nn.Flatten(-2, -1), nn.Linear(embed_dim, output_dim))

    def forward(
        self,
        node_feats: Float[Tensor, "V d_v"],
        edge_feats: Float[Tensor, "E d_e"],
        edge_index: Int[Tensor, "2 E"],
    ) -> Float[Tensor, "V d_o"]:
        src, dest = edge_index.unbind(0)

        Q = self.W_q(node_feats[src])
        K = self.W_k(node_feats[dest])
        V = self.W_v(node_feats[dest])  # (E, h, d)
        bias = self.bias(edge_feats)

        scores = einsum("Ehd,Ehd->Eh", Q, K) / self.sqrt_dk + bias
        alpha = scatter_softmax(scores, dest, dim=0).unsqueeze(-1)
        H = scatter(alpha * V, dest, dim=0, dim_size=len(node_feats), reduce="sum")
        node_hiddens = self.W_o(H)

        return node_hiddens
