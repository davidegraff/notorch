from jaxtyping import Float
import torch
from torch import Tensor


def multilinear_inner_product(*tensors: Float[Tensor, "... d"]) -> Float[Tensor, "..."]:
    return torch.stack(tensors, dim=0).prod(0).sum(-1)


MIP = multilinear_inner_product
