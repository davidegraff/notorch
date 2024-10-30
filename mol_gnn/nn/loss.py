from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from mol_gnn.utils import ClassRegistry, ReprMixin


__all__ = [
    "LossFunction",
    "MSELoss",
    "BoundedMSELoss",
    "MVELoss",
    "EvidentialLoss",
    "BCELoss",
    "CrossEntropyLoss",
    "MccMixin",
    "BinaryMCCLoss",
    "MulticlassMCCLoss",
    "DirichletMixin",
    "BinaryDirichletLoss",
    "MulticlassDirichletLoss",
    "_ThresholdMixin",
    "SIDLoss",
    "WassersteinLoss",
]


class LossFunction(ABC, ReprMixin):
    """A :class:`LossFunction` calculates the fully reduced loss function given both a prediction and a target tensor"""

    def __call__(
        self,
        Y_hat: Tensor,
        Y: Tensor,
        mask: Tensor,
        sample_weights: Tensor,
        task_weights: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ):
        """Calculate the *fully reduced* loss value

        Parameters
        ----------
        Y_hat : Tensor
            a tensor of shape `b x t x *` containing the predictions
        Y : Tensor
            a tensor of shape `b x t` containing the target values
        mask : Tensor
            a boolean tensor of shape `b x t` indicating whether the given prediction should be
            included in the loss calculation
        sample_weights : Tensor
            a tensor of shape `b` or `b x 1` containing the per-sample weight
        task_weights : Tensor
            a tensor of shape `t` or `1 x t` containing the per-task weight
        lt_mask: Tensor
        gt_mask: Tensor

        Returns
        -------
        Tensor
            a scalar containing the fully reduced loss
        """
        L = self.forward(Y_hat, Y, mask, sample_weights, task_weights, lt_mask, gt_mask)
        L = L * sample_weights.view(-1, 1) * task_weights.view(1, -1) * mask

        return L.sum() / mask.sum()

    @abstractmethod
    def forward(
        self, Y_hat, Y, mask, sample_weights, task_weights, lt_mask, gt_mask
    ) -> Tensor:
        """Calculate the *unreduced* loss tensor."""


LossFunctionRegistry = ClassRegistry[LossFunction]()


@LossFunctionRegistry.register("mse")
class MSELoss(LossFunction):
    def forward(self, Y_hat: Tensor, Y: Tensor, *args) -> Tensor:
        return F.mse_loss(Y_hat, Y, reduction="none")


@LossFunctionRegistry.register("bounded-mse")
class BoundedMSELoss(MSELoss):
    def forward(
        self,
        Y_hat: Tensor,
        Y: Tensor,
        mask,
        sample_weights,
        task_weights,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ) -> Tensor:
        Y_hat = torch.where((Y_hat < Y) & lt_mask, Y, Y_hat)
        Y_hat = torch.where((Y_hat > Y) & gt_mask, Y, Y_hat)

        return super().forward(Y_hat, Y)


@LossFunctionRegistry.register("mve")
class MVELoss(LossFunction):
    """Calculate the loss using Eq. 9 from [nix1994]_

    References
    ----------
    .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
        probability distribution." Proceedings of 1994 IEEE International Conference on Neural
        Networks, 1994 https://doi.org/10.1109/icnn.1994.374138
    """

    def forward(self, Y_hat: Tensor, Y: Tensor, *args) -> Tensor:
        mean, var = torch.chunk(Y_hat, 2, 1)

        L_nll = (mean - Y) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2

        return L_nll + L_kl


@LossFunctionRegistry.register("evidential")
class EvidentialLoss(LossFunction):
    """Caculate the loss using Eq. **TODO** from [soleimany2021]_

    References
    ----------
    .. [soleimany2021] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.;
        "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." ACS
        Cent. Sci. 2021, 7, 8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546
    """

    def __init__(self, v_kl: float = 0.2, eps: float = 1e-8):
        self.v_kl = v_kl
        self.eps = eps

    def forward(self, Y_hat: Tensor, Y: Tensor, *args) -> Tensor:
        mean, v, alpha, beta = torch.chunk(Y_hat, 4, 1)

        residuals = Y - mean
        twoBlambda = 2 * beta * (1 + v)

        L_nll = (
            0.5 * (torch.pi / v).log()
            - alpha * twoBlambda.log()
            + (alpha + 0.5) * torch.log(v * residuals**2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        L_reg = (2 * v + alpha) * residuals.abs()

        return L_nll + self.v_kl * (L_reg - self.eps)

    def get_params(self) -> list[tuple[str, float]]:
        return [("v_kl", self.v_kl), ("eps", self.eps)]


@LossFunctionRegistry.register("bce")
class BCELoss(LossFunction):
    def forward(self, Y_hat: Tensor, Y: Tensor, *args) -> Tensor:
        return F.binary_cross_entropy_with_logits(Y_hat, Y, reduction="none")


@LossFunctionRegistry.register("ce")
class CrossEntropyLoss(LossFunction):
    def forward(self, Y_hat: Tensor, Y: Tensor, *args) -> Tensor:
        Y_hat = Y_hat.transpose(1, 2)
        Y = Y.long()

        return F.cross_entropy(Y_hat, Y, reduction="none")


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
        *args
    ):
        if not (0 <= Y_hat.min() and Y_hat.max() <= 1):  # assume logits
            Y_hat = Y_hat.softmax(2)

        L = self.forward(Y_hat, Y.long(), mask, sample_weights, *args)
        L = L * task_weights

        return L.mean()


@LossFunctionRegistry.register("binary-mcc")
class BinaryMCCLoss(LossFunction, MccMixin):
    def forward(self, Y_hat, Y, mask, sample_weights, *args) -> Tensor:
        TP = (Y * Y_hat * sample_weights * mask).sum(0, keepdim=True)
        FP = ((1 - Y) * Y_hat * sample_weights * mask).sum(0, keepdim=True)
        TN = ((1 - Y) * (1 - Y_hat) * sample_weights * mask).sum(0, keepdim=True)
        FN = (Y * (1 - Y_hat) * sample_weights * mask).sum(0, keepdim=True)

        MCC = (TP * TN - FP * FN) / (
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        ).sqrt()

        return 1 - MCC


@LossFunctionRegistry.register("multiclass-mcc")
class MulticlassMCCLoss(LossFunction, MccMixin):
    def forward(self, Y_hat, Y, mask, sample_weights, *args) -> Tensor:
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

        # the `einsum` calls amount to calculating the batched dot product
        cov_ytyp = c * s - torch.einsum("ij,ij->i", p, t).sum()
        cov_ypyp = s2 - torch.einsum("ij,ij->i", p, p).sum()
        cov_ytyt = s2 - torch.einsum("ij,ij->i", t, t).sum()

        x = cov_ypyp * cov_ytyt
        MCC = torch.tensor(0.0, device=device) if x == 0 else cov_ytyp / x.sqrt()

        return 1 - MCC


class DirichletMixin:
    """Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

    References
    ----------
    .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
        classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function
    """

    def __init__(self, v_kl: float = 0.2):
        self.v_kl = v_kl

    def forward(self, Y_hat, Y, *args) -> Tensor:
        S = Y_hat.sum(-1, keepdim=True)
        p = Y_hat / S

        A = (Y - p).square().sum(-1, keepdim=True)
        B = ((p * (1 - p)) / (S + 1)).sum(-1, keepdim=True)

        L_mse = A + B

        alpha = Y + (1 - Y) * Y_hat
        beta = torch.ones_like(alpha)
        S_alpha = alpha.sum(-1, keepdim=True)
        S_beta = beta.sum(-1, keepdim=True)

        ln_alpha = S_alpha.lgamma() - alpha.lgamma().sum(-1, keepdim=True)
        ln_beta = beta.lgamma().sum(-1, keepdim=True) - S_beta.lgamma()

        dg0 = torch.digamma(alpha)
        dg1 = torch.digamma(S_alpha)

        L_kl = (
            ln_alpha
            + ln_beta
            + torch.sum((alpha - beta) * (dg0 - dg1), -1, keepdim=True)
        )

        return (L_mse + self.v_kl * L_kl).mean(-1)

    def get_params(self) -> list[tuple[str, float]]:
        return [("v_kl", self.v_kl)]


@LossFunctionRegistry.register("binary-dirichlet")
class BinaryDirichletLoss(DirichletMixin, LossFunction):
    def forward(self, Y_hat: Tensor, Y: Tensor, *args) -> Tensor:
        N_CLASSES = 2
        n_tasks = Y.shape[1]
        Y_hat = Y_hat.reshape(len(Y_hat), n_tasks, N_CLASSES)
        y_one_hot = torch.eye(N_CLASSES, device=Y_hat.device)[Y.long()]

        return super().forward(Y_hat, y_one_hot, *args)


@LossFunctionRegistry.register("multiclass-dirichlet")
class MulticlassDirichletLoss(DirichletMixin, LossFunction):
    def forward(self, Y_hat: Tensor, Y: Tensor, mask: Tensor, *args) -> Tensor:
        y_one_hot = torch.eye(Y_hat.shape[2], device=Y_hat.device)[Y.long()]

        return super().forward(Y_hat, y_one_hot, mask)


@dataclass
class _ThresholdMixin:
    threshold: float | None = None

    def get_params(self) -> list[tuple[str, float]]:
        return [("threshold", self.threshold)]


@LossFunctionRegistry.register("sid")
class SIDLoss(LossFunction, _ThresholdMixin):
    def forward(self, Y_hat: Tensor, Y: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            Y_hat = Y_hat.clamp(min=self.threshold)

        preds_norm = Y_hat / (Y_hat * mask).sum(1, keepdim=True)

        Y = Y.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (preds_norm / Y).log() * preds_norm + (Y / preds_norm).log() * Y


@LossFunctionRegistry.register(["earthmovers", "wasserstein"])
class WassersteinLoss(LossFunction, _ThresholdMixin):
    def forward(self, Y_hat: Tensor, Y: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            Y_hat = Y_hat.clamp(min=self.threshold)

        preds_norm = Y_hat / (Y_hat * mask).sum(1, keepdim=True)

        return (Y.cumsum(1) - preds_norm.cumsum(1)).abs()
