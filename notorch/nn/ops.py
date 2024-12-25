import torch
from torch import Tensor
import torch.nn as nn


class _OpBase(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return f"dim={self.dim}"


class Add(_OpBase):
    """Add the input tensors along :attr:`dim`

    Parameters
    ----------
    dim : int, default=-1
        the dimension along which to sum
    """

    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.stack(tensors, dim=self.dim).sum(dim=self.dim)


class Prod(_OpBase):
    """Multiply the input tensors along :attr:`dim`

    Parameters
    ----------
    dim : int, default=-1
        the dimension along which to multiply
    """

    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.stack(tensors, dim=self.dim).prod(dim=self.dim)


class Cat(_OpBase):
    """Concatenate the input tensors along :attr:`dim`.

    Parameters
    ----------
    dim : int, default=-1
        the dimension along which to concatenate
    """

    def __init__(self, dim: int = -1):
        super().__init__(dim)

    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.cat(tensors, dim=self.dim)


class Split(nn.Module):
    """Split the input tensor into chunks of :attr:`split_size` along :attr:`dim`.

    See also
    --------
    - :func:`torch.split`
    """

    def __init__(self, split_size: int = 1, dim: int = -1):
        super().__init__()

        self.split_size = split_size
        self.dim = dim

    def forward(self, tensor: Tensor) -> tuple[Tensor, ...]:
        return torch.split(tensor, self.split_size, self.dim)


class MatMul(nn.Module):
    """Multiply the two matrices :attr:`A` and :attr:`B`.

    Parameters
    ----------
    transpose : bool, default False
        whether to transpose the last two dimensions of :attr:`B`

    See also
    --------
    - :func:`torch.matmul`
    """

    def __init__(self, transpose: bool = False) -> None:
        super().__init__()

        self.transpose = transpose

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        if self.transpose:
            B = B.mT

        return A @ B

    def extra_repr(self) -> str:
        return f"transpose={self.transpose}"


class Einsum(nn.Module):
    def __init__(self, equation: str):
        super().__init__()

        self.equation = equation

    def foward(self, *tensors: Tensor) -> Tensor:
        return torch.einsum(self.equation, *tensors)

    def extra_repr(self) -> str:
        return f"equation={repr(self.equation)}"
