from dataclasses import dataclass

import numpy as np
from mol_gnn.featurizers.graph.base import GraphFeaturizer

from mol_gnn.types import Mol
from mol_gnn.featurizers.graph.graph import Graph
from mol_gnn.featurizers.graph.mixins import _MolGraphFeaturizerMixin


@dataclass
class MolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    """A :class:`MolGraphFeaturizer` featurizes molecules into :class:`MolGraph`s`"""

    def __call__(self, mol: Mol) -> Graph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            V = np.zeros((1, self.node_dim))
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()])
        E = np.empty((2 * n_bonds, self.edge_dim))
        edge_index = [[], []]

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                e = self.bond_featurizer(bond)
                E[i : i + 2] = e

                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])

                i += 2

        rev_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return Graph(V, E, edge_index, rev_index)


@dataclass
class LineMolGraphFeaturizer(_MolGraphFeaturizerMixin, GraphFeaturizer[Mol]):
    def __post_init__(self):
        self.node_dim = self.node_dim + self.edge_dim

    def __call__(self, mol: Mol) -> Graph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        V = np.zeros((1, self.node_dim)) if n_atoms == 0 else np.empty((n_bonds, self.node_dim))
        E = []
        edge_index = [[], []]

        i = 0
        for i in range(n_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()

            self.add_edges_to_incident_bonds(edge_index, mol.GetAtomWithIdx(u), i)
            self.add_edges_to_incident_bonds(edge_index, mol.GetAtomWithIdx(v), i)

            x_v = self.atom_featurizer(mol.GetAtomWithIdx(u))
            x_u = self.atom_featurizer(mol.GetAtomWithIdx(v))
            x_e = self.bond_featurizer(bond)
            V[i] = np.concatenate([(x_u + x_v) / 2, x_e])

        rev_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return Graph(V, E, edge_index, rev_index)

    def add_edges_to_incident_bonds(self, edge_index, atom, i):
        for b in atom.GetBonds():
            j = b.GetIdx()
            if i == j:
                continue
            edge_index[0].extend([i])
            edge_index[1].extend([j])
