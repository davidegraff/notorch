from typing import NamedTuple

import numpy as np


class Graph(NamedTuple):
    """A :class:`Graph` represents the feature representation of graph."""

    V: np.ndarray
    """an array of shape ``V x d_v`` containing the vertex features of the graph"""
    E: np.ndarray
    """an array of shape ``E x d_e`` containing the edge features of the graph"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_index: np.ndarray
    """A array of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in :attr:`edge_index` attribute."""
