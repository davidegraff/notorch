from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph

from notorch.data.models.gvp import DualRankFeatures
from notorch.nn.rbf import RBFEmbedding
from notorch.nn.spatial.gvp.layers import Aggregation, Dropout, GatedGVP, LayerNorm
from notorch.types import Reduction


class EdgeEmbed(nn.Module):
    def __init__(self, d_min: float = 0, d_max: float = 4.5, num_bases: int = 16):
        super().__init__()

        self.rbf = RBFEmbedding(d_min, d_max, num_bases)

    @property
    def out_dims(self) -> tuple[int, int]:
        return self.rbf.num_bases, 1

    def forward(
        self, coords: Float[Tensor, "V r"], edge_index: Int[Tensor, "2 E"]
    ) -> Float[DualRankFeatures, "E num_bases", "E r 1"]:
        src, dest = edge_index.unbind(0)
        r_ij = coords[dest] - coords[src]
        d_ij = LA.vector_norm(r_ij, p=2, dim=-1)
        r_ij_norm = F.normalize(r_ij, dim=-1)

        scalar_feats = self.rbf(d_ij)
        vector_feats = r_ij_norm.unsqueeze(-1)

        return DualRankFeatures(scalar_feats, vector_feats)


class MessageFunction(nn.Module):
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        num_layers: int = 3,
        activations: tuple[type[nn.Module], type[nn.Module] | None] = (nn.ReLU, None),
    ):
        super().__init__()

        scalar_in_dim, vect_in_dim = in_dims
        scalar_edge_dim, vect_edge_dim = edge_dims
        message_in_dims = (2 * scalar_in_dim + scalar_edge_dim, 2 * vect_in_dim + vect_edge_dim)

        gvps = []
        if num_layers == 1:
            gvps = [GatedGVP(message_in_dims, out_dims, activations=(None, None))]
        else:
            gvps = (
                [GatedGVP(message_in_dims, out_dims)]
                + [
                    GatedGVP(out_dims, out_dims, activations=activations)
                    for _ in range(num_layers - 2)
                ]
                + [GatedGVP(out_dims, out_dims, activations=(None, None))]
            )

        self.gvp = nn.Sequential(*gvps)

    def forward(
        self,
        node_feats: Float[DualRankFeatures, "V d_s_in", "V r d_v_in"],
        edge_feats: Float[DualRankFeatures, "E d_s_in", "E r d_v_in"],
        edge_index: Int[Tensor, "2 E"],
    ) -> tuple[Float[Tensor, "E d_s_out"], Float[Tensor, "E  r d_v_out"]]:
        s, V = node_feats.scalar_feats, node_feats.vector_feats
        s_ij, V_ij = edge_feats.scalar_feats, edge_feats.vector_feats
        src, dest = edge_index.unbind(0)

        s = torch.cat([s[src], s[dest], s_ij], dim=-1)
        V = torch.cat([V[src], V[dest], V_ij], dim=-1)

        return self.gvp((s, V))


class GVPConv(nn.Module):
    def __init__(
        self,
        node_dims: tuple[int, int],
        radius: float = 4.5,
        edge_embed: EdgeEmbed | None = None,
        num_message_layers: int = 3,
        activations: tuple[type[nn.Module] | None, type[nn.Module] | None] = (nn.ReLU, None),
        dropout: float = 0.1,
        reduce: Reduction = "mean",
    ):
        edge_embed = edge_embed or EdgeEmbed()
        message = MessageFunction(
            node_dims, node_dims, edge_embed.out_dims, num_message_layers, activations, radius
        )

        self.edge_embed = edge_embed
        self.message = message
        self.radius = radius
        self.message = MessageFunction()
        self.dropout = Dropout(dropout)
        self.agg = Aggregation(reduce)
        self.layer_norm = LayerNorm(node_dims)

    def forward(
        self,
        node_feats: Float[DualRankFeatures, "V d_s_in", "V r d_v_in"],
        coords: Float[Tensor, "V r"],
        batch_index: Int[Tensor, "V"],
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        edge_index = radius_graph(coords, self.radius, batch_index)
        edge_feats = self.edge_embed(coords, edge_index)
        _, dest = edge_index.unbind(0)

        messages = self.dropout(self.message(node_feats, edge_feats, edge_index))
        agg_messages = DualRankFeatures(self.agg(messages, dest))
        hiddens = node_feats + agg_messages
        outputs = self.layer_norm((hiddens["scalar_feats"], hiddens["vector_feats"]))

        return outputs


class GVPGNNLayer(nn.Module):
    conv: GVPConv
    update: GatedGVP
    dropout: Dropout
    layer_norm: LayerNorm

    def __init__(
        self,
        node_dims: tuple[int, int],
        radius: float = 4.5,
        edge_embed: EdgeEmbed | None = None,
        num_message_layers: int = 3,
        num_update_layers: int = 2,
        activations: tuple[type[nn.Module] | None, type[nn.Module] | None] = (nn.ReLU, None),
        dropout: float = 0.1,
        reduce: Reduction = "mean",
    ):
        gvps = []
        if num_update_layers == 1:
            gvps = [GatedGVP(node_dims, node_dims, activations=(None, None))]
        else:
            scalar_dim, vector_dim = node_dims
            hidden_dims = (4 * scalar_dim, 2 * vector_dim)
            gvps = (
                [GatedGVP(node_dims, hidden_dims)]
                + [
                    GatedGVP(hidden_dims, hidden_dims, activations=activations)
                    for _ in range(num_update_layers - 2)
                ]
                + [GatedGVP(hidden_dims, node_dims, activations=(None, None))]
            )

        self.conv = GVPConv(
            node_dims, radius, edge_embed, num_message_layers, activations, dropout, reduce
        )
        self.update = nn.Sequential(*gvps)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(node_dims)

    def forward(
        self,
        node_feats: DualRankFeatures,
        coords: Float[Tensor, "V r"],
        batch_index: Int[Tensor, "V"],
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        node_feats = self.conv(node_feats, coords, batch_index)
        node_feats = node_feats + self.dropout(self.update(node_feats))
        node_feats = self.layer_norm(node_feats)

        return node_feats
