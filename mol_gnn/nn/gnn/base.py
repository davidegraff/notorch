from abc import abstractmethod
from typing import Annotated

from jaxtyping import Float
from torch import Tensor, nn

from mol_gnn.data.models.graph import BatchedGraph, Graph


class GNNLayer(nn.Module):
    @abstractmethod
    def forward(self, G: Graph) -> Graph:
        pass


class Aggregation(nn.Module):
    @abstractmethod
    def forward(
        self, G: Annotated[BatchedGraph, "(V d_v) (E d_e) b"], **kwargs
    ) -> Float[Tensor, "b d_v"]:
        pass
