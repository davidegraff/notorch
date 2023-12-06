from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass

import numpy as np
from rdkit import Chem

from mol_gnn.featurizers.molgraph.molgraph import MolGraph
from mol_gnn.featurizers.molgraph.mixins import _MolGraphFeaturizerMixin


class MoleculeMolGraphFeaturizer(ABC):
    """A :class:`MoleculeMolGraphFeaturizer` featurizes RDKit molecules into
    :class:`MolGraph`s"""

    @abstractmethod
    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        """Featurize the input molecule into a molecular graph

        Parameters
        ----------
        mol : Chem.Mol
            the input molecule
        atom_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated atom features
        bond_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated bond features

        Returns
        -------
        MolGraph
            the molecular graph of the molecule
        """


@dataclass
class BaseMoleculeMolGraphFeaturizer(MoleculeMolGraphFeaturizer, _MolGraphFeaturizerMixin):
    """A :class:`BaseMoleculeMolGraphFeaturizer` is the base implementation of a
    :class:`MoleculeMolGraphFeaturizer`"""

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        # if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
        #     raise ValueError(
        #         "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
        #         f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
        #     )
        # if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
        #     raise ValueError(
        #         "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
        #         f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
        #     )

        if n_atoms == 0:
            V = np.zeros((1, self.atom_fdim))
        else:
            V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()])
        E = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                e = self.bond_featurizer(bond)
                if bond_features_extra is not None:
                    e = np.concatenate((e, bond_features_extra[bond.GetIdx()]))

                E[i : i + 2] = e

                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])

                i += 2

        rev_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(V, E, edge_index, rev_index)
