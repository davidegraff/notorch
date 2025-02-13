from jaxtyping import Float
from torch import Tensor
import torch


def cv(x: Tensor, dim: int = 0) -> Tensor:
    """compute the coefficient of variation of :attr:`x` along dimension :attr:`dim`."""
    return x.std(dim) / x.mean(dim)


def kth_excluding(A: Float[Tensor, "*b d"], k: int, *, beta: float = 1e6) -> Float[Tensor, "b d"]:
    r"""Calculate the :attr:`k`-th largest component of each row ``i`` when excluding column ``j``.

    More specifically, a matrix :math:`A \in \mathbb R^{n \times e}`, calculate each entry
    :math:`a_{ij}` as the :math:`k`-th largest element in the vector :math:`\mathbf a_i` when
    excluding the :math:`j`-th element of that vector.

    Parameters
    ----------
    A : Tensor
        a tensor of shape ``b x d``, where ``b`` is the batch dimension and ``d`` is the dimension
        that the function will be calculated along
    k : int
        the :math:`k` to consider when calculating each element
    beta : float, default=1e6
        a value to subtract along the diagonal when calculating the output matrix. This value
        must be larger than the largest range per-row range (
        :math:`\max_i (\max_j A_{ij} - \min_j A_{ij})`) or this function is not guaranteed to work.

    Raises
    ------
    ValueError
        if :attr:`k` is outside the range :math:`[0, d-1]`
    """
    if k >= A.shape[1]:
        raise ValueError(f"arg 'k' must be in the range [0, {A.shape[-1] - 1}]. got: {k}")

    A = torch.atleast_2d(A)
    I = torch.eye(A.shape[-1], device=A.device)
    B = A.unsqueeze(1) - beta * I

    return B.neg().kthvalue(k, dim=-1).values.neg()


coefficient_of_variation = cv
