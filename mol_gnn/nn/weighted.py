from typing import Iterable, Iterator, Literal, overload

from tensordict import TensorDict
import tensordict.nn as tdnn
import torch.nn as nn


class WeightedModuleList(nn.ModuleList):
    def __init__(self, modules: Iterable[nn.Module], weights: Iterable[float]) -> None:
        super().__init__(modules)

        self.weights = list(weights)

    def forward(self, td: TensorDict, mode: Literal["train", "val", "test"] = "train"):
        out_dict = {}
        loss = 0
        for module in self:
            td = module(td)

            name = module.out_keys[1]
            val = td[module.out_keys]

            out_dict[f"{mode}/{name}"] = val
            loss += self.loss_weights[name] * val

        return td

    @overload
    def __iter__(self) -> Iterator[tdnn.TensorDictModule]: ...
