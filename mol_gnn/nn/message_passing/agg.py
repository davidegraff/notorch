from abc import abstractmethod
from typing import Iterable

import torch
from torch import Tensor, nn
from torch_scatter import (
    scatter_max, scatter_mean, scatter_sum, scatter_softmax
)

from mol_gnn.conf import DEFAULT_MESSAGE_DIM
from mol_gnn.nn.permute import Permutation
from mol_gnn.utils import HasHParams
from mol_gnn.utils.registry import ClassRegistry


class Aggregation(nn.Module, HasHParams):
    def __init__(self, *, directed: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.directed = directed
        self.hparams = {
            "from_hparams": self.__class__,
            "directed": self.directed,
        }

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
        self.hparams.update({
            "input_dim": input_dim,
            "output_dim": output_dim,
            "slope": slope,
        })

        output_dim = input_dim if output_dim is None else output_dim

        self.W_l = nn.Linear(input_dim, output_dim)
        self.W_r = nn.Linear(input_dim, output_dim)
        self.act = nn.LeakyReLU(slope)
        self.A = nn.Linear(output_dim, 1)

    def _calc_weights(self, M: Tensor, dest: Tensor, rev_index: Tensor) -> Tensor:
        H_l = self.W_l(M)
        H_r = self.W_r(M)
        H = H_l + H_r[rev_index]
        scores = self.A(self.act(H))

        return scatter_softmax(scores, dest, dim=0)


@AggregationRegistry.register("mlp-att")
class MLPAttention(_AttentionAggBase):
    def __init__(self, input_dim: int = DEFAULT_MESSAGE_DIM, *, directed: bool = True, **kwargs):
        super().__init__(directed=directed, **kwargs)
        self.hparams.update({
            "input_dim": input_dim,
        })

        self.A = nn.Linear(input_dim, 1)

    def _calc_weights(self, M: Tensor, dest: Tensor, rev_index: Tensor) -> Tensor:
        scores = self.A(M)

        return scatter_softmax(scores, dest, dim=0)


class MLP(Aggregation):
    """
    References
    ----------
    .. [1] arXiv:2211.04952v1 [cs.LG]. https://arxiv.org/pdf/2211.04952.pdf
    """
    def __init__(
        self,
        input_dim: int = DEFAULT_MESSAGE_DIM,
        max_num_nodes: int = 64,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.4,
        bias: bool = True,
        p: int = 0,
        *,
        directed: bool = True,
        **kwargs
    ):
        super().__init__(directed=directed, **kwargs)
        self.hparams.update({
            "input_dim": input_dim,
            "max_num_nodes": max_num_nodes,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "p": p,
        })

        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        output_dim = input_dim if output_dim is None else output_dim

        self.mlp = nn.Sequential(
            nn.Flatten(1, 2),
            nn.Linear(input_dim * max_num_nodes, hidden_dim, bias),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.max_num_nodes = max_num_nodes
        self.permute = MultiPermutation(1)
        self.p = p
    
    def gather(self, M: Tensor, dest: Tensor, rev_index: Tensor, dim_size: int | None) -> Tensor:
        M = self._pad_packed_graph(M, dest, self.max_num_nodes, dim_size)
        if self.p == 0:
            return self.mlp(M)
        
        Hs = []
        for _ in range(self.p):
            idxs = torch.randperm(self.max_num_nodes, device=M.device)
            Hs.append(self.mlp(M[:, idxs, :]))

        return torch.stack(Hs, dim=0).mean(0)

    @classmethod
    def _pad_packed_graph(cls, X: Tensor, batch: Tensor, max_size: int, dim_size: int | None):
        n_nodes = scatter_sum(torch.ones_like(batch), batch, dim=0, dim_size=dim_size)
        pad_sizes = max_size - n_nodes
        Xs = X.split(n_nodes.tolist())
        Xs_pad = [torch.nn.functional.pad(Xs[i], (0, 0, 0, pad_sizes[i])) for i in range(len(Xs))]

        return torch.stack(Xs_pad)
