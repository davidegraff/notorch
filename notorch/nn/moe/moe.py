import copy

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn

from notorch.nn.moe.routers import router


class MixtureOfExperts[T_mod: nn.Module](nn.Module):
    r"""

    Parameters
    ----------
    module : T_mod
        any learnable function :math:`f : \mathbb R^p \to \mathbb R_q`
    in_features : int
        the number of input features to the function
    num_experts : int
        the number of experts :math:`E`
    reset_parameters : bool, default=True
        whether to reset the parameters of the input module. Under the hood, this module
        calls :func:`copy.deepcopy` to generate the mixture. If ``False``, each expert will have
        identical starting weights.
    """
    def __init__(
        self,
        module: T_mod,
        in_features: int,
        num_experts: int,
        reset_parameters: bool = True,
        **kwargs,
    ):
        self.experts = nn.ModuleList([copy.deepcopy(module) for _ in num_experts])
        self.router = router(in_features, num_experts, **kwargs)

        if reset_parameters:
            self.experts.apply(
                lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
            )

    def forward(
        self, inputs: Float[Tensor, "b p"]
    ) -> tuple[Float[Tensor, "b q"], Float[Tensor, ""]]:
        outputss = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        weights, loss = self.router(inputs)
        output = (outputss * weights.unsqueeze(2)).sum(1)

        return output, loss
