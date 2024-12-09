from abc import abstractmethod
from typing import Annotated

from jaxtyping import Float
from torch import Tensor
import torch.nn as nn

from mol_gnn.data.models.graph import BatchedGraph, Graph


class GNNLayer(nn.Module):
    @abstractmethod
    def forward(
        self, G: Annotated[Graph, "(V t_v) (E t_e)"]
    ) -> Annotated[Graph, "(V d_v) (E d_e)"]:
        pass


class Aggregation(nn.Module):
    @abstractmethod
    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        pass
