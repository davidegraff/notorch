import copy

import torch.nn as nn

from notorch.nn.moe.routers import router


class MixtureOfExperts[T_mod: nn.Module](nn.Module):
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
