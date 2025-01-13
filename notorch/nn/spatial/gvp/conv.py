from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_scatter import scatter_mean

from notorch.nn.rbf import RBFEmbedding
from notorch.nn.spatial.gvp.layers import Dropout, GatedGVP, LayerNorm


class EdgeEmbed(nn.Module):
    def __init__(self, d_min: float = 0, d_max: float = 4.5, num_bases: int = 16):
        super().__init__()

        self.rbf = RBFEmbedding(d_min, d_max, num_bases)

    @property
    def out_dims(self) -> tuple[int, int]:
        return self.rbf.num_bases, 1

    def forward(
        self, coords: Float[Tensor, "V r"], edge_index: Int[Tensor, "2 E"]
    ) -> tuple[Float[Tensor, "E num_bases"], Float[Tensor, "E r 1"]]:
        src, dest = edge_index.unbind(0)
        r_ij = coords[dest] - coords[src]
        d_ij = LA.vector_norm(r_ij, p=2, dim=-1)
        r_ij_norm = F.normalize(r_ij, dim=-1)

        s_e = self.rbf(d_ij)
        V_e = r_ij_norm.unsqueeze(-1)

        return s_e, V_e


class MessageFunction(nn.Module):
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        edge_embed: EdgeEmbed,
        n_layers: int = 3,
        activations: tuple[type[nn.Module], type[nn.Module] | None] = (nn.ReLU, None),
        radius: float = 5,
    ):
        super().__init__()

        scalar_in_dim, vect_in_dim = in_dims
        scalar_edge_dim, vect_edge_dim = edge_embed.out_dims
        message_in_dims = (2 * scalar_in_dim + scalar_edge_dim, 2 * vect_in_dim + vect_edge_dim)

        modules = []
        if n_layers == 1:
            modules = [GatedGVP(message_in_dims, out_dims, activations=(None, None))]
        else:
            modules = (
                [GatedGVP(message_in_dims, out_dims)]
                + [
                    GatedGVP(out_dims, out_dims, activations=activations)
                    for _ in range(n_layers - 2)
                ]
                + [GatedGVP(out_dims, out_dims, activations=(None, None))]
            )

        self.edge_embed = edge_embed
        self.radius = radius
        self.mlp = nn.Sequential(*modules)

    def forward(
        self,
        node_feats: tuple[Float[Tensor, "V d_s_in"], Float[Tensor, "V r d_v_in"]],
        edge_feats: tuple[Float[Tensor, "E d_s_in"], Float[Tensor, "E r d_v_in"]],
        edge_index: Int[Tensor, "2 E"],
    ) -> tuple[Float[Tensor, "E d_s_out"], Float[Tensor, "E  r d_v_out"]]:
        s, V = node_feats
        s_ij, V_ij = edge_feats
        src, dest = edge_index.unbind(0)

        s = torch.cat([s[src], s[dest], s_ij], dim=-1)
        V = torch.cat([V[src], V[dest], V_ij], dim=-1)

        return self.mlp((s, V))


class GVPConv(nn.Module):
    edge_embed: EdgeEmbed
    message: MessageFunction
    dropout: Dropout
    layer_norm: LayerNorm
    radius: float

    def forward(
        self,
        node_feats: tuple[Float[Tensor, "V d_s"], Float[Tensor, "V r d_v"]],
        coords: Float[Tensor, "V r"],
        batch_index: Int[Tensor, "V"],
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        edge_index = radius_graph(coords, self.radius, batch_index)
        src, dest = edge_index.unbind(0)
        edge_feats = self.edge_embed(coords, edge_index)

        m_s_ij, m_V_ij = self.dropout(self.message(node_feats, edge_feats, edge_index))
        m_s = scatter_mean(m_s_ij, dest, dim=0, dim_size=len(s))
        m_V = scatter_mean(m_V_ij, dest, dim=0, dim_size=len(V))

        s, V = node_feats
        h_s = s + m_s
        h_v = V + m_V
        out = (h_s, h_v)

        return self.layer_norm(out)


class GVPGNNLayer(nn.Module):
    conv: GVPConv
    update: GatedGVP
    dropout: Dropout

    def forward(
        self,
        node_feats: tuple[Float[Tensor, "V d_s"], Float[Tensor, "V r d_v"]],
        coords: Float[Tensor, "V r"],
        batch_index: Int[Tensor, "V"],
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        node_feats = self.conv(node_feats, coords, batch_index)
        node_feats = self.dropout(self.update(node_feats))
    