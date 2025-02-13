from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn


class Add(nn.Module):
    """Add the input tensors element-wise."""

    def forward(self, *tensors: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return torch.stack(tensors, dim=0).sum(dim=0)


class Mul(nn.Module):
    """Multiply the input tensors element-wise."""

    def forward(self, *tensors: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return torch.stack(tensors, dim=0).prod(dim=0)


class Cat(nn.Module):
    """Concatenate the input tensors along :attr:`dim`.

    Parameters
    ----------
    dim : int, default=-1
        the dimension along which to concatenate
    """

    def __init__(self, dim: int = -1):
        super().__init__()

        self.dim = dim

    def extra_repr(self):
        return f"dim={self.dim}"

    def forward(self, *tensors: Tensor) -> Tensor:
        return torch.cat(tensors, dim=self.dim)


class Split(nn.Module):
    """Split the input tensor into chunks of :attr:`split_size` along :attr:`dim`.

    Parameters
    ----------
    split_size : int, default=1
        the size of a single chunk along the given dimension
    dim : int, default=-1
        the dimension along which to split

    See Also
    --------
    :func:`torch.split`
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
    transpose : bool, default=False
        whether to transpose the last two dimensions of :attr:`B`

    See Also
    --------
    :func:`torch.matmul`
    """

    def __init__(self, transpose: bool = False) -> None:
        super().__init__()

        self.transpose = transpose

    def forward(
        self, A: Float[Tensor, "... n p"], B: Float[Tensor, "... p q"]
    ) -> Float[Tensor, "... n q"]:
        if self.transpose:
            B = B.mT

        return A @ B

    def extra_repr(self) -> str:
        return f"transpose={self.transpose}"


class Einsum(nn.Module):
    """Apply the given Einstein summation to the input tensors.

    Parameters
    ----------
    equation : str
        the Einstein summation to apply

    See Also
    --------
    :func:`torch.einsum`
    """

    def __init__(self, equation: str):
        super().__init__()

        self.equation = equation

    def foward(self, *tensors: Tensor) -> Tensor:
        return torch.einsum(self.equation, *tensors)

    def extra_repr(self) -> str:
        return f"equation={repr(self.equation)}"
