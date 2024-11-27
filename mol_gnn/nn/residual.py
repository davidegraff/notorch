from collections.abc import Callable
from typing import TypeVar

from torch import nn
import torch

T_mod = TypeVar("T_mod", bound=nn.Module)


class Residual[T_mod](nn.Module):
    def __init__(self, module: T_mod, op: Callable = torch.add):
        super().__init__()

        self.module = module
        self.op = op

    def forward(self, input):
        return self.op(input, self.module(input))
