from abc import abstractmethod
from typing import Iterable

import torch
from torch import Tensor, nn
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax

from mol_gnn.conf import DEFAULT_HIDDEN_DIM
from mol_gnn.nn.layers import MultiPermutation
from mol_gnn.utils.registry import ClassRegistry


class Aggregation(nn.Module):
    @abstractmethod
    def forward(
        self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """aggregate the messages at each node and return the aggregated message along with the weight
        assigned to each message during aggregation"""


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("sum")
class Sum(Aggregation):
    def forward(self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs) -> Tensor:
        M_v = scatter_sum(M, dest, dim=0, dim_size=dim_size)
        w = torch.ones(len(M), device=M.device).unsqueeze(1)

        return M_v, w


@AggregationRegistry.register("mean")
class Mean(Aggregation):
    def forward(self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs) -> Tensor:
        M_v = scatter_mean(M, dest, dim=0, dim_size=dim_size)
        w = torch.ones(len(M), device=M.device).unsqueeze(1)
        w = 1 / scatter_sum(w, dest, dim=0, dim_size=dim_size)[dest]

        return M_v, w


@AggregationRegistry.register("max")
class Max(Aggregation):
    BETA = 100

    def forward(self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs) -> Tensor:
        M_v = scatter_max(M, dest, dim=0, dim_size=dim_size)
        w = scatter_softmax(M * self.BETA, dest, 0, dim_size=dim_size).unsqueeze(1)

        return M_v, w


class _AttentiveAggBase(Aggregation):
    def forward(self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs) -> Tensor:
        """forward the messages at each node in the graph"""
        alpha = self._calc_weights(M, dest, **kwargs)

        return scatter_sum(alpha * M, dest, dim=0, dim_size=dim_size), alpha

    @abstractmethod
    def _calc_weights(self, M: Tensor, dest: Tensor, **kwargs) -> Tensor:
        """Calculate the attention weights for message aggregation.

        Parameters
        ----------
        M : Tensor
            A tensor of shape ``M x d`` containing message feature matrix.
        dest : Tensor
            a tensor of shape ``M`` containing the destination of each message
        **kwargs
            additional keyword arguments

        Returns
        -------
        Tensor
            a tensor of shape ``M`` containing the attention weights.
        """


@AggregationRegistry.register("gatv2")
class GATv2(_AttentiveAggBase):
    """something resemembling GATv2 but not quite"""

    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int | None = None,
        slope: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.hparams.update({"input_dim": input_dim, "output_dim": output_dim, "slope": slope})

        output_dim = input_dim if output_dim is None else output_dim

        self.W_0 = nn.Linear(input_dim, output_dim)
        self.W_1 = nn.Linear(input_dim, output_dim)
        self.act = nn.LeakyReLU(slope)
        self.a = nn.Linear(output_dim, 1)

    def _calc_weights(self, M: Tensor, dest: Tensor, rev_index: Tensor) -> Tensor:
        Z = self.act(self.W_0(M) + self.W_1(M[rev_index]))
        scores = self.a(Z)

        return scatter_softmax(scores, dest, dim=0)


@AggregationRegistry.register("gated")
class GatedAttention(_AttentiveAggBase):
    """A learnable aggregation gate"""

    def __init__(self, input_dim: int = DEFAULT_HIDDEN_DIM, **kwargs):
        super().__init__(**kwargs)
        # self.hparams.update({"input_dim": input_dim})

        self.a = nn.Linear(input_dim, 1)

    def _calc_weights(self, M: Tensor, dest: Tensor, **kwargs) -> Tensor:
        scores = self.a(M)

        return scatter_softmax(scores, dest, dim=0)


class MLP(Aggregation):
    """
    References
    ----------
    .. [1] arXiv:2211.04952v1 [cs.LG]. https://arxiv.org/pdf/2211.04952.pdf
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        max_num_nodes: int = 64,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.4,
        bias: bool = True,
        p: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hparams.update(
            {
                "input_dim": input_dim,
                "max_num_nodes": max_num_nodes,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "p": p,
            }
        )

        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        output_dim = input_dim if output_dim is None else output_dim

        self.mlp = nn.Sequential(
            nn.Flatten(1, 2),
            nn.Linear(input_dim * max_num_nodes, hidden_dim, bias),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.max_num_nodes = max_num_nodes
        self.permute = MultiPermutation(1)
        self.p = p

    def forward(self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs) -> Tensor:
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


class CompositeAggregation(Aggregation):
    aggs: list[Aggregation]

    def __init__(self, aggs: Iterable[Aggregation]):
        super().__init__()

        self.aggs = nn.ModuleList(aggs)

    @property
    def mult(self) -> int:
        """the multiplier to apply to the output dimension"""
        return len(self.aggs)

    def forward(self, M: Tensor, dest: Tensor, dim_size: int | None, **kwargs) -> Tensor:
        Ms, _ = zip(*[agg(M, dest, dim_size, **kwargs) for agg in self.aggs])

        return torch.cat(Ms, dim=1)


class MessageAggregation(nn.Module):
    def __init__(self, agg: Aggregation):
        super().__init__()

        self.agg = agg

    @property
    def mult(self) -> int:
        """the multiplier to apply to the output dimension"""
        return 1

    @abstractmethod
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        """Aggregate the incoming messages"""


class NodeAggregation(MessageAggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        _, dest = edge_index

        return self.agg(M, dest, rev_index, dim_size)[0]


class EdgeAggregation(MessageAggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        src, dest = edge_index

        M = (M + M[rev_index]) / 2
        M_v, _ = self.agg(M, dest, rev_index, dim_size)

        return M_v[src]


class DirectedEdgeAggregation(MessageAggregation):
    def forward(
        self, M: Tensor, edge_index: Tensor, rev_index: Tensor, dim_size: int | None
    ) -> Tensor:
        src, dest = edge_index

        M_v, w = self.agg(M, dest, rev_index, dim_size)

        return M_v[src] - w * M[rev_index]
