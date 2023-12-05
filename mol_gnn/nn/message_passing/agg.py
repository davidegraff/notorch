from abc import abstractmethod
from typing import Iterable

import torch
from torch import Tensor, nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax

from mol_gnn.conf import DEFAULT_MESSAGE_DIM
from mol_gnn.utils.registry import ClassRegistry


class Aggregation(nn.Module):
    def __init__(self, *args, directed: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.directed = directed

    @property
    def mult(self) -> int:
        """the multiplier to apply to the output dimension"""
        return 1
    
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        src, dest = edge_index
        M_v = self.gather(M, dest, rev_index, dim_size)

        return M_v[src] - M[rev_index] if self.directed else M_v[src]

    @abstractmethod
    def gather(self, M: Tensor, dest: Tensor, rev_index: Tensor, dim_size: int | None) -> Tensor:
        pass


AggregationRegistry = ClassRegistry[Aggregation]()


class CompositeAggregation(Aggregation):
    aggs: list[Aggregation]
    
    def __init__(self, aggs: Iterable[Aggregation]):
        super(Aggregation, self).__init__()

        self.aggs = nn.ModuleList(aggs)

    @property
    def mult(self) -> int:
        """the multiplier to apply to the output dimension"""
        return len(self.aggs)
    
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        Ms = [agg.forward(M, edge_index, rev_index, dim_size) for agg in self.aggs]
        
        return torch.cat(Ms, dim=1)

    def gather(self, M: Tensor, dest: Tensor, rev_index: Tensor, dim_size: int | None) -> Tensor:
        Ms = [agg.gather(M, dest, rev_index, dim_size) for agg in self.aggs]
        
        return torch.cat(Ms, dim=1)

@AggregationRegistry.register("sum")
class Sum(Aggregation):
    def gather(self, H: Tensor, dest: Tensor, rev_index: Tensor, dim_size: int | None) -> Tensor:
        return scatter_sum(H, dest, dim=0, dim_size=dim_size)


@AggregationRegistry.register("mean")
class Mean(Aggregation):
    def gather(self, H: Tensor, dest: Tensor, rev_index: Tensor, dim_size: int | None) -> Tensor:
        return scatter_mean(H, dest, dim=0, dim_size=dim_size)


@AggregationRegistry.register("max")
class Mean(Aggregation):
    def gather(self, H: Tensor, dest: Tensor, rev_index: Tensor, dim_size: int | None) -> Tensor:
        return scatter_max(H, dest, dim=0, dim_size=dim_size)


class _AttentionAggBase(Aggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        src, dest = edge_index

        alpha = self._calc_weights(M, dest, rev_index, dim_size)
        M_v = self.gather(M, dest, rev_index, dim_size, alpha)
        M = M_v[src]

        return M - (alpha * M)[rev_index] if self.directed else M

    def gather(
        self,
        M: Tensor,
        dest: Tensor,
        rev_index: Tensor,
        dim_size: int | None,
        alpha: Tensor | None = None,
    ) -> Tensor:
        """gather the messages at each node in the graph"""
        alpha = self._calc_weights(M, dest, rev_index, dim_size) if alpha is None else alpha

        return scatter_sum(alpha * M, dest, dim=0, dim_size=dim_size)

    @abstractmethod
    def _calc_weights(self, M: Tensor, dest: Tensor, rev_index: Tensor) -> Tensor:
        """Calculate the attention weights for message aggregation.

        Parameters
        ----------
        M : Tensor
            A tensor of shape ``M x d`` containing message feature matrix.
        dest : Tensor
            a tensor of shape ``M`` containing the destination of each message
        rev_index : Tensor
            a tensor of shape ``M`` containing the index of each message's reverse message

        Returns
        -------
        Tensor
            a tensor of shape ``M`` containing the attention weights.
        """


@AggregationRegistry.register("gatv2")
class GATv2(_AttentionAggBase):
    def __init__(
        self,
        input_dim: int = DEFAULT_MESSAGE_DIM,
        output_dim: int | None = None,
        slope: float = 0.2,
        *,
        directed: bool = True,
        **kwargs,
    ):
        super().__init__(directed=directed, **kwargs)

        output_dim = input_dim if output_dim is None else output_dim

        self.W_l = nn.Linear(input_dim, output_dim)
        self.W_r = nn.Linear(input_dim, output_dim)
        self.A = nn.Linear(output_dim, 1)
        self.act = nn.LeakyReLU(slope)

    def _calc_weights(self, M: Tensor, dest: Tensor, rev_index: Tensor) -> Tensor:
        H_l = self.W_l(M)
        H_r = self.W_r(M)
        H = H_l + H_r[rev_index]
        scores = self.A(self.act(H))

        return scatter_softmax(scores, dest, dim=0)


@AggregationRegistry.register("mlp")
class MLPAttention(_AttentionAggBase):
    def __init__(self, input_dim: int = DEFAULT_MESSAGE_DIM, *, directed: bool = True, **kwargs):
        super().__init__(directed=directed, **kwargs)

        self.A = nn.Linear(input_dim, 1)

    def _calc_weights(self, M: Tensor, dest: Tensor, rev_index: Tensor) -> Tensor:
        scores = self.A(M)

        return scatter_softmax(scores, dest, dim=0)
