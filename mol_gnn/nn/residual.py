from collections.abc import Callable
from typing import TypeVar

from torch import nn
import torch

T = TypeVar("T", bound=nn.Module)


class Residual[T](nn.Module):
    def __init__(self, module: T, op: Callable = torch.add):
        super().__init__()

        self.module = module
        self.op = op

    def forward(self, input):
        return self.op(input, self.module(input))
