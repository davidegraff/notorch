from dataclasses import dataclass

import numpy as np
from mol_gnn.featurizers.molgraph.base import MolGraphFeaturizer

from mol_gnn.types import Mol
from mol_gnn.featurizers.molgraph.molgraph import MolGraph
from mol_gnn.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin


@dataclass
class SimpleMoleculeMolGraphFeaturizer(_MolGraphFeaturizerMixin, MolGraphFeaturizer[Mol]):
    """A :class:`SimpleMoleculeMolGraphFeaturizer` featurizes molecules into :class:`MolGraph`s`"""

    def __call__(self, mol: Mol) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim))
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()])
        E = np.empty((2 * n_bonds, self.bond_fdim))
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

        return MolGraph(V, E, edge_index, rev_index)
