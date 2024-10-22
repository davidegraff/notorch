from dataclasses import dataclass

import networkx as nx
import numpy as np
from mol_gnn.featurizers.graph.base import GraphFeaturizer

from mol_gnn.types import Mol
from mol_gnn.data.models.graph import Graph
from mol_gnn.featurizers.graph.mixins import _MolGraphFeaturizerMixin


def calc_rev_index(edge_index: np.ndarray) -> np.ndarray:
    edge_index = edge_index.T
    rev_mask = np.all(edge_index[None, :] == np.flip(edge_index, axis=-1)[:, None], axis=-1)

    return np.where(rev_mask)[1]


@dataclass
class MolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`MolGraphFeaturizer` featurizes molecules into :class:`MolGraph`s`"""

    def __call__(self, mol: Mol) -> Graph:
        if mol.GetNumAtoms() == 0:
            V = np.zeros((1, self.node_dim))
            E = np.zeros((0, self.edge_dim))
            edge_index = np.zeros((2, 0), int)

            return Graph(V, E, edge_index, None)

        G = self.mol_to_nx_graph(mol).to_directed()

        V = np.stack(list(nx.get_node_attributes(G, "x").values()), dtype=float)
        edges, E = zip(*nx.get_edge_attributes(G, "x").items())
        E = np.stack(E, dtype=float)
        edge_index = np.stack(edges, dtype=int).T
        # sink = np.array(list(nx.get_node_attributes(G, 'sink').values()), int).T

        return Graph(V, E, edge_index, calc_rev_index(edge_index))

    def mol_to_nx_graph(self, mol) -> nx.Graph:
        n_atoms = mol.GetNumAtoms()

        G = nx.Graph()
        for u in range(n_atoms):
            a = mol.GetAtomWithIdx(u)
            G.add_node(u, x=self.atom_featurizer(a), sink=[u])
            for v in range(u + 1, n_atoms):
                b = mol.GetBondBetweenAtoms(u, v)
                if b is None:
                    continue
                G.add_edge(u, v, x=self.bond_featurizer(b))

        return G
