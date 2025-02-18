from abc import abstractmethod

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn


__all__ = ["RankNContrastLoss"]


class CDist(nn.Module):
    @abstractmethod
    def forward(
        self, A: Float[Tensor, "*b n d"], B: Float[Tensor, "*b m d"] | None = None
    ) -> Float[Tensor, "n m"]:
        pass


class PNorm(CDist):
    def __init__(self, p: float = 2.0, negate: bool = False) -> None:
        super().__init__()

        self.p = p
        self.negate = negate

    def forward(self, A: Tensor, B: Tensor | None = None) -> Tensor:
        if B is None:
            B = A

        X = torch.cdist(A, B, p=self.p)

        return X.neg() if self.negate else X

    def extra_repr(self) -> str:
        return f"p={self.p:0.1f}, negate={self.negate}"


class RankNContrastLoss(nn.Module):
    EPS: float = 1e-6

    def __init__(
        self, distance: CDist | None = None, similarity: CDist | None = None, temp: float = 2.0
    ):
        super().__init__()

        self.distance = distance or PNorm(p=1)
        self.similarity = similarity or PNorm(p=2, negate=True)
        self.temp = temp

    def forward(
        self, inputs: Float[Tensor, "b d"], targets: Float[Tensor, "b t"]
    ) -> Float[Tensor, ""]:
        N = len(targets)
        dists = self.distance(targets)
        sims = self.similarity(inputs)
        scores = torch.exp(sims / self.temp)

        mask = dists[..., None, :] >= dists[..., :, None]
        off_diag = ~torch.eye(N, dtype=bool)
        off_diag_index = off_diag.argwhere().unbind(1)
        i, j, k = (mask & off_diag.unsqueeze(-1)).argwhere().unbind(1)
        # mapping from a given k to index of its unique (i, j) pair
        _, k_to_ij = torch.unique_consecutive(j, return_inverse=True)

        scores_ij = scores[off_diag_index]
        sum_scores_ik = (
            torch.zeros_like(scores_ij)
            .scatter_reduce_(0, k_to_ij, scores[i, k], reduce="sum")
            .add(self.EPS)
        )
        probs_ij = scores_ij / sum_scores_ik
        nlls_ji = probs_ij.log().neg()
        # losses_i = nlls_ji.reshape(N, N - 1).mean(-1)

        return nlls_ji.mean()

    def extra_repr(self) -> str:
        return f"(temp): {self.temp:0.1f}"
