from abc import abstractmethod

from jaxtyping import Bool, Float
from numpy.typing import ArrayLike
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class _LossFunctionBase(nn.Module):

    @abstractmethod
    def forward(
        self,
        preds: Float[Tensor, "b t ..."],
        targets: Float[Tensor, "b t ..."],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
    ) -> Float[Tensor, ""]:
        pass

    def _reduce(
        self,
        loss: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"] | None = None,
        sample_weights: Float[Tensor, "b"] | None = None,
    ) -> Float[Tensor, ""]:
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(0)

        return loss.mean() if mask is None else (loss * mask).sum() / mask.sum()


class _BoundedMixin:
    def forward(
        self,
        preds: Float[Tensor, "b t"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
        lt_mask: Bool[Tensor, "b t"],
        gt_mask: Bool[Tensor, "b t"],
    ) -> Float[Tensor, ""]:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)

        return super().forward(preds, targets, mask, sample_weights)


class MSE(_LossFunctionBase):
    def forward(
        self,
        preds: Float[Tensor, "b t"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] | None = None,
        sample_weights: Float[Tensor, "b"] | None = None,
    ) -> Float[Tensor, ""]:
        L = F.mse_loss(preds, targets, reduction="none")

        return self._reduce(L, mask, sample_weights)


class BoundedMSE(_BoundedMixin, MSE):
    pass


class MeanVarianceEstimation(_LossFunctionBase):
    """Calculate the loss using Eq. 9 from [nix1994]_

    References
    ----------
    .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
        probability distribution." Proceedings of 1994 IEEE International Conference on Neural
        Networks, 1994 https://doi.org/10.1109/icnn.1994.374138
    """

    def forward(
        self,
        preds: Float[Tensor, "b t 2"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        mean, var = torch.unbind(preds, dim=-1)

        L_nll = (mean - targets) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2
        L = L_nll + L_kl

        return self._reduce(L, mask, sample_weights)


class Evidential(_LossFunctionBase):
    """Caculate the loss using Eq. **TODO** from [soleimany2021]_

    References
    ----------
    .. [soleimany2021] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.;
        "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." ACS
        Cent. Sci. 2021, 7, 8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546
    """

    def __init__(self, v_kl: float = 0.2, eps: float = 1e-8):
        super().__init__()

        self.v_kl = v_kl
        self.eps = eps

    def forward(
        self,
        preds: Float[Tensor, "b t 4"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
    ) -> Float[Tensor, ""]:
        mean, v, alpha, beta = torch.unbind(preds, dim=-1)

        residuals = targets - mean
        twoBlambda = 2 * beta * (1 + v)

        L_nll = (
            0.5 * (torch.pi / v).log()
            - alpha * twoBlambda.log()
            + (alpha + 0.5) * torch.log(v * residuals**2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        L_reg = (2 * v + alpha) * residuals.abs()
        L = L_nll + self.v_kl * (L_reg - self.eps)

        return self._reduce(L, mask, sample_weights)

    def extra_repr(self) -> str:
        return f"v_kl={self.v_kl:0.1f}, eps={self.eps:0.1e}"


class BinaryCrossEntropy(_LossFunctionBase):
    def forward(
        self,
        preds: Float[Tensor, "b t"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
    ) -> Float[Tensor, ""]:
        L = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")

        return self._reduce(L, mask, sample_weights)


class CrossEntropy(_LossFunctionBase):
    def forward(
        self,
        preds: Float[Tensor, "b t k"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
    ) -> Float[Tensor, ""]:
        preds = preds.transpose(1, 2)
        targets = targets.long()
        L = F.cross_entropy(preds, targets, reduction="none")

        return self._reduce(L, mask, sample_weights)


'''
class MccMixin:
    """Calculate a soft `Matthews correlation coefficient`_ loss for multiclass
    classification based on the `Scikit-Learn implementation`_

    .. _Matthews correlation coefficient:
        https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case
    .. _Scikit-Learn implementation:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    def __call__(
        self,
        Y_hat: Tensor,
        Y: Tensor,
        mask: Tensor,
        sample_weights: Tensor,
        task_weights: Tensor,
        *args,
    ):
        if not (0 <= Y_hat.min() and Y_hat.max() <= 1):  # assume logits
            Y_hat = Y_hat.softmax(2)

        L = self._forward(Y_hat, Y.long(), mask, sample_weights, *args)
        L = L * task_weights

        return L.mean()


class BinaryMCCLoss(LossFunctionBase, MccMixin):
    def _forward_unreduced(self, Y_hat, Y, mask, sample_weights, *args) -> Tensor:
        TP = (Y * Y_hat * sample_weights * mask).sum(0, keepdim=True)
        FP = ((1 - Y) * Y_hat * sample_weights * mask).sum(0, keepdim=True)
        TN = ((1 - Y) * (1 - Y_hat) * sample_weights * mask).sum(0, keepdim=True)
        FN = (Y * (1 - Y_hat) * sample_weights * mask).sum(0, keepdim=True)

        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)).sqrt()

        return 1 - MCC


class MulticlassMCCLoss(LossFunctionBase, MccMixin):
    def _forward_unreduced(self, Y_hat, Y, mask, sample_weights, *args) -> Tensor:
        device = Y_hat.device

        C = Y_hat.shape[2]
        bin_targets = torch.eye(C, device=device)[Y]
        bin_preds = torch.eye(C, device=device)[Y_hat.argmax(-1)]
        masked_data_weights = sample_weights.unsqueeze(2) * mask.unsqueeze(2)

        p = (bin_preds * masked_data_weights).sum(0)
        t = (bin_targets * masked_data_weights).sum(0)
        c = (bin_preds * bin_targets * masked_data_weights).sum()
        s = (Y_hat * masked_data_weights).sum()
        s2 = s.square()

        cov_ytyp = c * s - torch.einsum("ij,ij->i", p, t).sum()
        cov_ypyp = s2 - torch.einsum("ij,ij->i", p, p).sum()
        cov_ytyt = s2 - torch.einsum("ij,ij->i", t, t).sum()

        x = cov_ypyp * cov_ytyt
        MCC = torch.tensor(0.0, device=device) if x == 0 else cov_ytyp / x.sqrt()

        return 1 - MCC
'''


class Dirichlet(_LossFunctionBase):
    """Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

    References
    ----------
    .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
        classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function
    """

    def __init__(self, v_kl: float = 0.2) -> None:
        super().__init__()

        self.v_kl = v_kl

    def forward(
        self,
        alphas: Float[Tensor, "b t k"],
        targets: Float[Tensor, "b t"],
        *,
        mask: Bool[Tensor, "b t"] = None,
        sample_weights: Float[Tensor, "b"] = None,
        **kwargs,
    ) -> Float[Tensor, ""]:
        alphas = F.softplus(alphas) + 1
        targets = F.one_hot(targets, num_classes=2)

        S = alphas.sum(-1, keepdim=True)
        probs = alphas / S
        A = (targets - probs).square().sum(-1)
        B = ((probs * (1 - probs)) / (S + 1)).sum(-1)
        L_mse = A + B

        alpha_tilde = targets + (1 - targets) * alphas
        beta = torch.ones_like(alpha_tilde)
        S_alpha = alpha_tilde.sum(-1)
        S_beta = beta.sum(-1)
        ln_alpha = S_alpha.lgamma() - alpha_tilde.lgamma().sum(-1)
        ln_beta = beta.lgamma().sum(-1) - S_beta.lgamma()
        dg0 = torch.digamma(alpha_tilde)
        dg1 = torch.digamma(S_alpha).unsqueeze(-1)
        L_kl = ln_alpha + ln_beta + ((alpha_tilde - beta) * (dg0 - dg1)).sum(-1)

        L = L_mse + self.v_kl * L_kl

        return self._reduce(L, mask, sample_weights)

    def extra_repr(self) -> str:
        return f"v_kl={self.v_kl:0.1f}"


"""
@dataclass
class _ThresholdMixin:
    threshold: float | None = None


class SIDLoss(LossFunctionBase, _ThresholdMixin):
    def _forward_unreduced(self, Y_hat: Tensor, Y: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            Y_hat = Y_hat.clamp(min=self.threshold)

        preds_norm = Y_hat / (Y_hat * mask).sum(1, keepdim=True)

        Y = Y.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (preds_norm / Y).log() * preds_norm + (Y / preds_norm).log() * Y


class WassersteinLoss(LossFunctionBase, _ThresholdMixin):
    def _forward_unreduced(self, Y_hat: Tensor, Y: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            Y_hat = Y_hat.clamp(min=self.threshold)

        preds_norm = Y_hat / (Y_hat * mask).sum(1, keepdim=True)

        return (Y.cumsum(1) - preds_norm.cumsum(1)).abs()
"""

BCE = BinaryCrossEntropy
XENT = CrossEntropy
MVE = MeanVarianceEstimation
