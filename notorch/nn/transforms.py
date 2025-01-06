from typing import get_args

from jaxtyping import Float
from numpy.typing import ArrayLike
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from notorch.exceptions import InvalidChoiceError
from notorch.types import TaskTransformConfig, TaskType


class _AffineBase(nn.Module):
    loc: Float[Tensor, "t"]
    scale: Float[Tensor, "t"]

    def __init__(self, loc: Float[ArrayLike, "t"], scale: Float[ArrayLike, "t"]) -> None:
        super().__init__()

        self.register_buffer("loc", torch.as_tensor(loc))
        self.register_buffer("scale", torch.as_tensor(scale))

    def extra_repr(self) -> str:
        return f"loc={self.loc}, scale={self.scale}"


class Normalize(_AffineBase):
    def forward(self, input: Float[Tensor, "b t"]) -> Float[Tensor, "b t"]:
        return (input - self.loc) / self.scale


class InverseNormalize(_AffineBase):
    def forward(self, input: Float[Tensor, "b t"]) -> Float[Tensor, "b t"]:
        return input * self.scale + self.loc


class MVE(_AffineBase):
    def forward(self, input: Float[Tensor, "b t 2"]) -> Float[Tensor, "b t 2"]:
        mean, var = input.unbind(-1)

        mean = mean * self.scale + self.loc
        var = var * self.scale**2

        return torch.stack([mean, var], -1)


class Evidential(_AffineBase):
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


def build(task_type: TaskType | None, values: Float[Tensor, "n t"]) -> TaskTransformConfig:
    if task_type is None:
        preds_transform = None
        target_transform = None
    elif task_type in ["regression", "mve", "evidential"]:
        mean = values.mean(0)
        std = values.std(0)
        match task_type:
            case "regression":
                preds_transform = InverseNormalize(mean, std)
            case "mve":
                preds_transform = MVE(mean, std)
            case "evidential":
                preds_transform = Evidential(mean, std)
        target_transform = Normalize(mean, std)
    else:
        match task_type:
            case "classification":
                preds_transform = nn.Sigmoid()
            case "multiclass":
                preds_transform = nn.Softmax(-1)
            case "dirichlet":
                preds_transform = Dirichlet()
            case _:
                raise InvalidChoiceError(task_type, get_args(TaskType))
        target_transform = None

    return {"preds": preds_transform, "targets": target_transform}
