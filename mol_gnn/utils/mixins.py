from collections.abc import Collection
import inspect
from typing import Any

from jaxtyping import Num
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


class CollateNDArrayMixin:
    def collate(self, inputs: Collection[Num[NDArray, "d"]]) -> Num[Tensor, "n d"]:
        return torch.from_numpy(np.array(inputs)).to(torch.float)


class ReprMixin:
    def __repr__(self) -> str:
        items = self.get_params()

        if len(items) > 0:
            keys, values = zip(*items)
            sig = inspect.signature(self.__class__)
            defaults = [sig.parameters[k].default for k in keys]
            items = [(k, v) for k, v, d in zip(keys, values, defaults) if v != d]

        argspec = ", ".join(f"{k}={repr(v)}" for k, v in items)

        return f"{self.__class__.__name__}({argspec})"

    def get_params(self) -> Collection[tuple[str, Any]]:
        return self.__dict__.items()
