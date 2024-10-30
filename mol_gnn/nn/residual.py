from collections.abc import Callable
from typing import TypeVar

from torch import nn
import torch

T_mod = TypeVar("T_mod", bound=nn.Module)


class Residual[S_inp, T_mod](nn.Module):
    def __init__(self, module: T_mod, op: Callable[[S_inp, S_inp], S_inp] = torch.add):
        super().__init__()

        self.module = module
        self.op = op

    def forward(self, input: S_inp):
        return self.op(input, self.module(input))
