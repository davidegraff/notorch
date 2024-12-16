from typing import Any, Callable, get_args

from jaxtyping import Float
from numpy.typing import ArrayLike
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mol_gnn.exceptions import InvalidChoiceError
from mol_gnn.types import TaskType


class _AffineTransform(nn.Module):
    loc: Float[Tensor, "d"]
    scale: Float[Tensor, "d"]

    def __init__(self, loc: Float[ArrayLike, "*d"], scale: Float[ArrayLike, "*d"]) -> None:
        super().__init__()

        self.register_buffer("loc", torch.as_tensor(loc))
        self.register_buffer("scale", torch.as_tensor(scale))

    def extra_repr(self) -> str:
        return f"loc={self.loc}, scale={self.scale}"


class Normalize(_AffineTransform):
    def forward(self, input: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return (input - self.loc) / self.scale


class InverseNormalize(_AffineTransform):
    def forward(self, input: Float[Tensor, "b d"]) -> Float[Tensor, "b d"]:
        return input * self.scale + self.loc


class MVE(_AffineTransform):
    def forward(self, input: Float[Tensor, "b t 2"]) -> Float[Tensor, "b t 2"]:
        mean, var = input.unbind(-1)

        mean = mean * self.scale + self.loc
        var = var * self.scale**2

        return torch.stack([mean, var], -1)


class Evidential(_AffineTransform):
    def forward(self, input: Float[Tensor, "b t 4"]) -> Float[Tensor, "b t 4"]:
        mean, var, alpha, beta = input.unbind(-1)

        var = F.softplus(var)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)

        mean = self.scale * mean + self.loc
        var = var * self.scale**2

        return torch.stack([mean, var, alpha, beta], -1)


class Dirichlet(nn.Module):
    def forward(self, input: Float[Tensor, "b t k"]) -> Float[Tensor, "b t (k+1)"]:
        k = input.shape[-1]
        alpha = F.softplus(input) + 1
        S = alpha.sum(-1, keepdim=True)

        return torch.cat([alpha / S, k / S], dim=-1)


class TransformManager(nn.Module):
    def __init__(self, transform: Callable[..., Any]):
        super().__init__()

        self.transform = transform

    def forward(self, input):
        return input if self.training else self.transform(input)


def build(
    task: TaskType, targets: Float[Tensor, "n t"]
) -> tuple[nn.Module | None, nn.Module | None]:
    match task:
        case "regression" | "mve" | "evidential":
            loc = targets.mean(0)
            scale = targets.std(0)
            match task:
                case "regression":
                    model_transform = InverseNormalize(loc, scale)
                case "mve":
                    model_transform = MVE(loc, scale)
                case "evidential":
                    model_transform = Evidential(loc, scale)
            target_transform = Normalize(loc, scale)
        case "dirichlet":
            model_transform = Dirichlet()
            target_transform = None
        case "classification":
            model_transform = nn.Sigmoid()
            target_transform = None
        case "multiclass":
            model_transform = nn.Softmax(-1)
            target_transform = None
        case _:
            raise InvalidChoiceError(task, get_args(TaskType))

    return (model_transform, target_transform)
