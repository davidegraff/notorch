from torch import Tensor


from typing import NamedTuple


class Graph(NamedTuple):
    """A :class:`Graph` represents the feature representation of graph."""

    V: Tensor
    """an array of shape ``V x d_v`` containing the vertex features of the graph"""
    E: Tensor
    """an array of shape ``E x d_e`` containing the edge features of the graph"""
    edge_index: Tensor
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_index: Tensor
    """A array of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in :attr:`edge_index` attribute."""

    @property
    def A(self) -> Tensor:
        pass