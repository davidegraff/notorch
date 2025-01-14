from jaxtyping import Float
import torch
from torch import Tensor
import torch.linalg as LA
import torch.nn as nn


class GatedEquivariantBlock(nn.Module):
    """The Gated Equivariant Block from [1]_

    Parameters
    ----------
    in_dims : tuple[int, int]
        the dimensionality of the scalar and vector inputs, respectively.
    out_dims : tuple[int, int]
        the dimensionality of the scalar and vector outputs, respectively.
    hidden_dim : int | None, default=None
        the hidden scalar dimension. If ``None``, will be calulated as
        ``out_dims[1] + max(in_dims[0], out_dims[1])``.
    act : type[nn.Module], default=nn.SiLU
        the activation to apply to the hidden scalar features.

    References
    ----------
    .. [1] arXiv:2102.03150v4 [cs.LG]
    """

    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        hidden_dim: int | None = None,
        act: type[nn.Module] = nn.SiLU,
    ):
        scalar_in_dim, vector_in_dim = in_dims
        scalar_out_dim, vector_out_dim = out_dims
        if hidden_dim is None:
            hidden_dim = vector_out_dim + max(scalar_in_dim, scalar_out_dim)

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.W_u = nn.Linear(vector_in_dim, scalar_in_dim, bias=False)
        self.W_v = nn.Linear(vector_in_dim, vector_out_dim, bias=False)
        self.W_s = nn.Sequential(
            nn.Linear(2 * scalar_in_dim, hidden_dim), act(), nn.Linear(hidden_dim, sum(out_dims))
        )

    def forward(
        self, inputs: tuple[Float[Tensor, "V d_s_in"], Float[Tensor, "V r d_v_in"]]
    ) -> tuple[Float[Tensor, "V d_s_out"], Float[Tensor, "V r d_v_out"]]:
        s, V = inputs

        V_u = self.W_u(V)
        v_v = LA.vector_norm(self.W_v(V), p=2, dim=-2)

        s_h = torch.cat([s, v_v], dim=-1)
        s_o, v_g = torch.split(self.W_s(s_h), self.out_dims, dim=-1)
        V_o = V_u * v_g.unsqueeze(1)

        return s_o, V_o


GEB = GatedEquivariantBlock
