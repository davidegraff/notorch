from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
import torch.linalg as LA


class GeometricVectorPerceptron(nn.Module):
    """not sure why this is in here. Prefer :class:`GatedGVP`"""
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        hidden_dim: int | None = None,
        acts: tuple[type[nn.Module], type[nn.Module]] = (nn.ReLU, nn.Sigmoid),
    ):
        super().__init__()

        scalar_in_dim, vect_in_dim = in_dims
        scalar_out_dim, vect_out_dim = out_dims
        scalar_act, vector_act = acts
        if hidden_dim is None:
            hidden_dim = max(vect_in_dim, vect_out_dim)

        self.W_h = nn.Linear(vect_in_dim, hidden_dim, bias=False)
        self.W_m = nn.Linear(scalar_in_dim + hidden_dim, scalar_out_dim)
        self.W_u = nn.Linear(hidden_dim, vect_out_dim, bias=False)
        self.scalar_act = scalar_act()
        self.vector_act = vector_act() if vector_act is not None else None


    def forward(
        self, inputs: tuple[Float[Tensor, "V d_s_in"], Float[Tensor, "V r d_v_in"]]
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        s, V = inputs

        V_h = self.W_h(V)
        v_h = LA.vector_norm(V_h, p=2, dim=-2)
        V_u = self.W_u(V_h)
        v_u = LA.vector_norm(V_u, p=2, dim=-2)
        s_m = self.W_m(torch.cat([s, v_h]))

        s_o = self.scalar_act(s_m)
        V_o = V_u * self.vector_act(v_u.unsqueeze(-2))

        return s_o, V_o


class GatedGeometricVectorPerceptron(nn.Module):
    """The updated GVP implementaion detailed in [1]_

    Parameters
    ----------
    in_dims : tuple[int, int]
        the dimensionality of the scalar and vector inputs, respectively.
    out_dims : tuple[int, int]
        the dimensionality of the scalar and vector outputs, respectively.
    hidden_dim : int | None, default=None
        the hidden vector dimension.
    acts : tuple[type[nn.Module] | None, type[nn.Module] | None], default=(nn.ReLU, None)
        the activations to apply to the scalar output and input to the vector gating, respectively.

    References
    ----------
    .. [1] arXiv:2106.03843 [cs.LG]
    """
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        hidden_dim: int | None = None,
        acts: tuple[type[nn.Module] | None, type[nn.Module] | None] = (nn.ReLU, None),
    ):
        super().__init__()

        scalar_in_dim, vect_in_dim = in_dims
        scalar_out_dim, vect_out_dim = out_dims
        scalar_act, vector_act = acts
        if hidden_dim is None:
            hidden_dim = max(vect_in_dim, vect_out_dim)

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.W_h = nn.Linear(vect_in_dim, hidden_dim, bias=False)
        self.W_u = nn.Linear(hidden_dim, vect_out_dim, bias=False)
        self.W_m = nn.Linear(scalar_in_dim + hidden_dim, scalar_out_dim)
        self.W_g = nn.Linear(scalar_out_dim, vect_out_dim)
        self.scalar_act = scalar_act()
        self.vector_act = vector_act() if vector_act is not None else None
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, inputs: tuple[Float[Tensor, "V d_s_in"], Float[Tensor, "V r d_v_in"]]
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        s, V = inputs

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
