from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
import torch.linalg as LA


class GeometricVectorPerceptron(nn.Module):
    """not sure why this is in here. Prefer :class:`GatedGVP`"""
    def __init__(
        self,
        in_features: tuple[int, int],
        out_features: tuple[int, int],
        hidden_dim: int | None = None,
        acts: tuple[type[nn.Module], type[nn.Module]] = (nn.ReLU, nn.Sigmoid),
    ):
        super().__init__()

        scalar_in_feats, vect_in_feats = in_features
        scalar_out_feats, vect_out_feats = out_features
        scalar_act, vector_act = acts
        if hidden_dim is None:
            hidden_dim = max(vect_in_feats, vect_out_feats)

        self.W_h = nn.Linear(vect_in_feats, hidden_dim, bias=False)
        self.W_m = nn.Linear(scalar_in_feats + hidden_dim, scalar_out_feats)
        self.W_u = nn.Linear(hidden_dim, vect_out_feats, bias=False)
        self.scalar_act = scalar_act()
        self.vector_act = vector_act() if vector_act is not None else None


    def forward(
        self, s: Float[Tensor, "V d_s_in"], V: Float[Tensor, "V r d_v_in"]
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        V_h = self.W_h(V)
        v_h = LA.vector_norm(V_h, p=2, dim=-2)
        V_u = self.W_u(V_h)
        v_u = LA.vector_norm(V_u, p=2, dim=-2)
        s_m = self.W_m(torch.cat([s, v_h]))

        s_o = self.scalar_act(s_m)
        V_o = V_u * self.vector_act(v_u.unsqueeze(-2))

        return s_o, V_o


class GatedGeometricVectorPerceptron(nn.Module):
    def __init__(
        self,
        in_features: tuple[int, int],
        out_features: tuple[int, int],
        hidden_dim: int | None = None,
        acts: tuple[type[nn.Module], type[nn.Module] | None] = (nn.ReLU, None),
    ):
        super().__init__()

        scalar_in_feats, vect_in_feats = in_features
        scalar_out_feats, vect_out_feats = out_features
        scalar_act, vector_act = acts
        if hidden_dim is None:
            hidden_dim = max(vect_in_feats, vect_out_feats)

        self.W_h = nn.Linear(vect_in_feats, hidden_dim, bias=False)
        self.W_m = nn.Linear(scalar_in_feats + hidden_dim, scalar_out_feats)
        self.W_u = nn.Linear(hidden_dim, vect_out_feats, bias=False)
        self.W_g = nn.Linear(scalar_out_feats, vect_out_feats)
        self.scalar_act = scalar_act()
        self.vector_act = vector_act() if vector_act is not None else None
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, s: Float[Tensor, "V d_s_in"], V: Float[Tensor, "V r d_v_in"]
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        V_h = self.W_h(V)
        V_u = self.W_u(V_h)
        v_h = LA.vector_norm(V_h, p=2, dim=-2)
        s_m = self.W_m(torch.cat([s, v_h]))

        if self.vector_act is not None:
            v_g = self.W_g(s_m)
        else:
            v_g = self.W_g(self.vector_act(s_m))

        s_o = self.scalar_act(s_m)
        V_o = V_u * self.sigmoid(v_g.unsqueeze(-2))

        return s_o, V_o


GVP = GeometricVectorPerceptron
GatedGVP = GatedGeometricVectorPerceptron
