from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


class _Residual[T](nn.Module):
    def __init__(self, module: T, op: Callable = torch.add):
        super().__init__()

        self.module = module
        self.op = op

    def forward(self, input):
        return self.op(input, self.module(input))


class Residual[T](nn.Module):
    def __init__(self, module: T):
        super().__init__()

        self.module = module

    def forward(self, *inputs):
        return inputs[0] + self.module(*inputs)
