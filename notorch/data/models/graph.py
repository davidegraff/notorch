from dataclasses import InitVar, dataclass, field
import textwrap
from typing import Iterable, Self

from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.types import Device

from notorch.conf import REPR_INDENT


@dataclass(repr=False, eq=False)
class Graph:
    """A :class:`Graph` represents the feature representation of graph."""

    node_feats: Int[Tensor, "V t_v"]
    """a tensor of shape ``|V| x t_v`` containing the vertex types/features of the graph"""
    edge_feats: Int[Tensor, "E t_e"]
    """a tensor of shape ``|E| x t_e`` containing the edge types/features of the graph"""
    edge_index: Int[Tensor, "2 E"]
    """a tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_index: Int[Tensor, "edge_feats"]
    """a tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in :attr:`edge_index` attribute."""
    device_: InitVar[Device] = field(default=None, kw_only=True)

    def __post_init__(self, device_: Device):
        self.__device = device_
        self.to(device_)

    @property
    def num_nodes(self) -> int:
        return len(self.node_feats)

    @property
    def num_edges(self) -> int:
        return len(self.edge_feats)

    @property
    def device(self) -> Device:
        return self.__device

    def to(self, device: Device) -> Self:
        self.__device = device

        self.node_feats = self.node_feats.to(device)
        self.edge_feats = self.edge_feats.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_index = self.rev_index.to(device)

        return self

    @property
    def A(self) -> Int[Tensor, "V V"]:
        """The dense adjacency matrix."""
        num_nodes = self.node_feats.shape[0]
        src, dest = self.edge_index.unbind(0)

        A = torch.zeros(num_nodes, num_nodes)
        A[src, dest] = 1

        return A

    @property
    def P(self) -> Float[Tensor, "V V"]:
        """The markov transition matrix."""
        A = self.A
        P = A / A.sum(1, keepdim=True)

        return P

    @property
    def dense2sparse(self) -> Int[Tensor, "V V"]:
        """A tensor of shape ``|V| x |V|`` mapping a dense edge index to its index in the edge
        features tensor.

        Because the the :class:`Graph` uses a sparse edge representation, this tensor allows
        one to index an edge by its corresponding nodes::

            # get the features of the edges going from 0->1 and 2->1
            G: Graph
            sparse_edge_index = G.dense2sparse[[0, 2], [1, 1]]
            G.E[sparse_edge_index].shape
            # (2, d)
        """
        num_nodes = self.node_feats.shape[0]
        src, dest = self.edge_index.unbind(0)

        index = -torch.ones(num_nodes, num_nodes, dtype=torch.long)
        index[src, dest] = torch.arange(self.edge_index.shape[1])

        return index

    def random_walk(
        self,
        length: int,
        num_walks: int = 1,
        starting_nodes: Tensor | None = None,
        return_edge_ids: bool = True,
    ) -> tuple[Int[Tensor, "n w l-1"], Int[Tensor, "n w l"] | None]:
        """Generate a random walk trace of given length from the starting nodes.

        Parameters
        ----------
        length : int
            the length of each walk
        num_walks : int, default=1
            The number of walks to start from each node in :attr:`starting_nodes`.
        starting_nodes : Tensor | None, default=None
            A tensor of shape ``n`` containing the index of the starting nodes in each walk. If
            ``None``, will start a walk at each node in the graph.
        return_edge_ids : bool, default=True
            Whether to return the edge IDs traversed in each walk.

        Returns
        -------
        Tensor
            a tensor of shape ``n x w x (l + 1)`` containing the ID of each node in the walk, where
            ``n`` is the number of starting nodes, ``w`` is the number of walks to start from each
            node, and ``l`` is the length of the walk.
        Tensor | None
            a tensor of shape ``n x w x l`` containing the ID of each edge in the walk if
            :attr:`return_edge_ids` is ``True``. Otherwise, ``None``
        """
        num_nodes = len(self.node_feats)

        if starting_nodes is None:
            starting_nodes = torch.arange(num_nodes)

        P = self.P
        node_ids = [starting_nodes.view(-1, 1).repeat(1, num_walks)]
        for _ in range(length):
            curr_node_ids = node_ids[-1]
            pi = torch.zeros(num_nodes, num_nodes).scatter_(1, curr_node_ids, value=1)
            pi = pi / pi.sum(-1, keepdim=True)
            new_node_ids = (pi @ P).multinomial(num_walks, replacement=True)
            node_ids.append(new_node_ids)

        node_ids: Tensor = torch.stack(node_ids, -1)
        if return_edge_ids:
            edge_ids = self.dense2sparse[node_ids[..., :-1], node_ids[..., 1:]]
        else:
            edge_ids = None

        return (node_ids, edge_ids)

    def __repr__(self) -> str:
        lines = (
            [f"{self.__class__.__name__}("]
            + [textwrap.indent(line, REPR_INDENT) for line in self._build_field_info()]
            + [")"]
        )

        return "\n".join(lines)

    def _build_field_info(self) -> list[str]:
        return [
            f"node_feats: Tensor(shape={self.node_feats.shape})",
            f"edge_feats: Tensor(shape={self.edge_feats.shape})",
            f"device={self.__device}",
            "",
        ]


@dataclass(repr=False, eq=False, kw_only=True)
class BatchedGraph(Graph):
    """A :class:`BatchedMolGraph` represents a batch of individual :class:`Graph`s."""

    batch_node_index: Int[Tensor, "V"]
    """A tensor of shape ``V`` containing the index of the parent :class:`Graph` of each node the
    batched graph."""
    batch_edge_index: Int[Tensor, "E"]
    """A tensor of shape ``E`` containing the index of the parent :class:`Graph` of each edge the
    batched graph."""
    size: InitVar[int | None] = None
    """The number of graphs, if known. Otherwise, will be estimated via
    :code:`batch_node_index.max() + 1`"""

    def __post_init__(self, device_: torch.device | str | int | None, size: int | None):
        super().__post_init__(device_)

        self.__size = self.batch_node_index.max() + 1 if size is None else size

    @classmethod
    def from_graphs(cls, Gs: Iterable[Graph]):
        node_featss = []
        edge_featss = []
        edge_indices = []
        rev_indices = []
        batch_node_indices = []
        batch_edge_indices = []
        offset = 0

        for i, G in enumerate(Gs):
            node_featss.append(G.node_feats)
            edge_featss.append(G.edge_feats)
            edge_indices.append(G.edge_index + offset)
            rev_indices.append(G.rev_index + offset)
            batch_node_indices.extend([i] * len(G.node_feats))
            batch_edge_indices.extend([i] * len(G.edge_feats))

            offset += len(G.node_feats)

        node_feats = torch.cat(node_featss, dim=0)
        edge_feats = torch.cat(edge_featss, dim=0)
        edge_index = torch.cat(edge_indices, dim=1).long()
        rev_index = torch.cat(rev_indices, dim=0).long()
        batch_node_index = torch.tensor(batch_node_indices, dtype=torch.long)
        batch_edge_index = torch.tensor(batch_edge_indices, dtype=torch.long)
        size = i + 1

        return cls(
            node_feats,
            edge_feats,
            edge_index,
            rev_index,
            device_=G.device,
            batch_node_index=batch_node_index,
            batch_edge_index=batch_edge_index,
            size=size,
        )

    def __len__(self) -> int:
        """The number of individual :class:`Graph`s in this batch"""
        return self.__size

    def to(self, device: Device) -> Self:
        self.__device = device

        self.node_feats = self.node_feats.to(device)
        self.edge_feats = self.edge_feats.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_index = self.rev_index.to(device)
        self.batch_node_index = self.batch_node_index.to(device)
        self.batch_edge_index = self.batch_edge_index.to(device)

        return self

    def _build_field_info(self) -> list[str]:
        return [
            f"node_feats: Tensor(shape={self.node_feats.shape})",
            f"edge_feats: Tensor(shape={self.edge_feats.shape})",
            f"device={self.__device}",
            f"batch_size={len(self)}" "",
        ]
