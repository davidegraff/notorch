from collections.abc import Collection

from jaxtyping import Num
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


class CollateNDArrayMixin:
    def collate(self, inputs: Collection[Num[NDArray, "d"]]) -> Num[Tensor, "n d"]:
        return torch.from_numpy(np.array(inputs)).to(torch.float)
