import torch
from torch import Tensor, nn


class Permutation(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()

        if dim < 1:
            raise ValueError(f"arg 'dim' must be greater than! got: {dim}")
        
        self.dim = dim

    def forward(self, X: Tensor):
        batch_perms = torch.stack(
            [torch.randperm(X.shape[self.dim]) for _ in range(len(X))]
        )
        index = batch_perms.unsqueeze(X.ndim - self.dim).expand(X.shape)

        return X.gather(self.dim, index)

class MultiPermutation(nn.Module):
    def __init__(self, permute: Permutation, n_permutations: int = 0):
        super().__init__()

        self.permute = permute
        self.n_permutations = n_permutations

    def forward(self, X: Tensor):
        return [self.permute(X) for _ in range(self.n_permutations)]