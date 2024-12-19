from torch import Tensor
import torch
import torch.nn as nn


class _OpBase(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return f"dim={self.dim}"


class Add(_OpBase):
    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.stack(tensors, dim=self.dim).sum(dim=self.dim)


class Prod(_OpBase):
    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.stack(tensors, dim=self.dim).prod(dim=self.dim)


class Cat(_OpBase):
    def __init__(self, dim: int = -1):
        super().__init__(dim)

    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.cat(tensors, dim=self.dim)
