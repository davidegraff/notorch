from jaxtyping import Float
from numpy.typing import ArrayLike
import torch
from torch import Tensor, nn


class _AffineTransformBase(nn.Module):
    def __init__(self, loc: Float[ArrayLike, "*d"], scale: Float[ArrayLike, "*d"]) -> None:
        super().__init__()

        self.loc = torch.as_tensor(loc)
        self.scale = torch.as_tensor(scale)

    def extra_repr(self) -> str:
        return f"loc={self.loc}, scale={self.scale}"


class AffineTransform(_AffineTransformBase):
    def forward(self, X: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return (X - self.loc) / self.scale


class InverseAffineTransform(_AffineTransformBase):
    def forward(self, X: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return X * self.scale + self.loc
