from abc import abstractmethod
from jaxtyping import Float
from torch import Tensor
import torch
import torch.distributions as D
import torch.nn as nn

from notorch.nn.utils import cv, kth_excluding


class KeepTopK(nn.Module):
    def __init__(self, k: int, dim: int = -1, beta: float = 1e6):
        super().__init__()

        self.k = k
        self.dim = dim
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        values, indices = x.topk(self.k, dim=self.dim, largest=True)

        return torch.full_like(x, self.beta).scatter_(1, indices, values)

    def extra_repr(self) -> str:
        return f"k={self.k}, dim={self.dim}, beta={self.beta:0.0e}"


# def keep_top_k(x: Tensor, k: int, dim: int = -1, beta: float = 1e6) -> Tensor:
#     values, indices = x.topk(k, dim, largest=True)

#     return torch.full_like(x, beta).scatter_(1, indices, values)


class Router(nn.Module):
    """A :class:`Router` calculates routing weights for a mixture of experts and an auxiliary loss.

    Parameters
    ----------
    in_features : int
        the number of input features to the router
    num_experts : int
        the number of experts :math:`E`
    """

    @abstractmethod
    def __init__(self, in_features: int, num_experts: int, **kwargs): ...

    @abstractmethod
    def forward(
        self, x: Float[Tensor, "b d"]
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, ""]]: ...


class DenseRouter(Router):
    def __init__(self, in_features: int, num_experts: int, *, v_imp: float = 0.1):
        super().__init__()

        self.G = nn.Sequential(nn.Linear(in_features, num_experts, bias=False), nn.Softmax(dim=-1))
        self.v_imp = v_imp

    def forward(self, x: Float[Tensor, "b d"]) -> tuple[Float[Tensor, "b e"], Float[Tensor, ""]]:
        g = self.G(x)
        l_imp = cv(g.sum(dim=0)) ** 2

        return g, self.v_imp * l_imp


class SparseRouter(Router):
    def __init__(
        self,
        in_features: int,
        num_experts: int,
        *,
        k: int = 1,
        noisy: bool = True,
        v_imp: float = 0.1,
        v_load: float = 0.1,
    ):
        super().__init__()

        if not (1 <= k < num_experts):
            raise ValueError(f"arg 'k' must be in the range [0, {num_experts - 1}] ! got: {k}.")

        self.W_g = nn.Linear(in_features, num_experts, bias=False)
        self.W_s = nn.Sequential(nn.Linear(in_features, num_experts, bias=False), nn.Softplus())
        self.noise = D.Normal(0, 1) if noisy else None
        self.G = nn.Sequential(KeepTopK(k), nn.Softmax(dim=-1))

        self.k = k
        self.v_imp = v_imp
        self.v_load = v_load

    def forward(self, x: Float[Tensor, "b d"]) -> tuple[Float[Tensor, "b e"], Float[Tensor, ""]]:
        g = self.W_g(x)
        if self.noise is not None:
            scale = self.W_s(x)
            h = g + self.noise.sample((len(x), 1)) * scale
            P = self.noise.cdf((g - kth_excluding(h, self.k)) / scale)
            load = P.sum(dim=0)
            l_load = cv(load) ** 2
        else:
            h = g
            l_load = 0

        l_imp = cv(h.sum(dim=0)) ** 2
        l_aux = self.v_imp * l_imp + self.v_load * l_load

        return self.G(h), l_aux


def router(
    in_features: int,
    num_experts: int,
    k: int | None = None,
    noisy: bool = True,
    v_imp: float = 0.1,
    v_load: float = 0.1,
) -> Router:
    if k is None or k == num_experts:
        return DenseRouter(in_features, num_experts, v_imp=v_imp)
    else:
        return SparseRouter(in_features, num_experts, k=k, noisy=noisy, v_imp=v_imp, v_load=v_load)
