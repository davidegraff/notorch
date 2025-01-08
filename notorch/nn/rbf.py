from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn


class RBFEmbedding(nn.Module):
    means: Float[Tensor, "1 n_bases"]

    def __init__(self, d_min: float, d_max: float, num_bases: int):
        super().__init__()

        means = torch.linspace(d_min, d_max, num_bases)
        width = (d_min - d_max) / num_bases

        self.register_buffer("means", means.unsqueeze(0))
        self.width = 0.5 * 1 / width**2

    def forward(self, D: Float[Tensor, "b"]) -> Float[Tensor, "b n_bases"]:
        diffs = D.unsqueeze(-1) - self.means

        return torch.exp(-self.factor *  diffs**2)

    def extra_repr(self) -> str:
        return f"d_min={self.means[0]}, d_max={self.means[-1]}, num_bases={len(self.means)}"
