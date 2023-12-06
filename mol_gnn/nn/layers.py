import torch
from torch import Tensor, nn


class Permutation(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()

        if dim < 1:
            raise ValueError(f"arg 'dim' must be greater than! got: {dim}")

        self.dim = dim

    def forward(self, X: Tensor):
        batched_perm_idxs = torch.stack([torch.randperm(X.shape[self.dim]) for _ in range(len(X))])
        index = batched_perm_idxs.unsqueeze(X.ndim - self.dim).expand(X.shape)

        return X.gather(self.dim, index)


class MultiPermutation(nn.Module):
    def __init__(self, dim: int = 1, n_permutations: int = 0):
        super().__init__()

        self.permute = Permutation(dim)
        self.n_permutations = n_permutations

    def forward(self, X: Tensor) -> list[Tensor]:
        return [self.permute(X) for _ in range(self.n_permutations)]


class Full(nn.Module):
    def __init__(self, output_dim: int, value: float = 0) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.value = value

    def forward(self, X: Tensor) -> Tensor:
        size = (*X.shape[:-1], self.output_dim)

        return torch.full(size, self.value, device=X.device)
