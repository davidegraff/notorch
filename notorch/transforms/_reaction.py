from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from enum import auto
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem

from notorch.data.models.graph import Graph
from notorch.transforms.base import TensorTransform
from notorch.types import Atom, Bond, Mol, Rxn
from notorch.utils.utils import EnumMapping


class RxnMode(EnumMapping):
    """The mode by which a reaction should be featurized into a `MolGraph`"""

    REAC_PROD = auto()
    """concatenate the reactant features with the product features."""
    REAC_PROD_BALANCE = auto()
    """concatenate the reactant features with the products feature and balances imbalanced
    reactions"""
    REAC_DIFF = auto()
    """concatenates the reactant features with the difference in features between reactants and
    products"""
    REAC_DIFF_BALANCE = auto()
    """concatenates the reactant features with the difference in features between reactants and
    product and balances imbalanced reactions"""
    PROD_DIFF = auto()
    """concatenates the product features with the difference in features between reactants and
    products"""
    PROD_DIFF_BALANCE = auto()
    """concatenates the product features with the difference in features between reactants and
    products and balances imbalanced reactions"""


@dataclass
class CondensedReactionGraphFeaturizer:
    """A :class:`CondensedGraphOfReactionFeaturizer` featurizes reactions using the condensed
    reaction graph method utilized in [1]_

    **NOTE**: This class *does not* accept a :class:`AtomFeaturizerProto` instance. This is because
    it requries the :meth:`num_only()` method, which is only implemented in the concrete
    :class:`AtomFeaturizer` class

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=AtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizerBase, default=BondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    mode_ : Union[str, ReactionMode], default=ReactionMode.REAC_DIFF
        the mode by which to featurize the reaction as either the string code or enum value

    References
    ----------
    .. [1] Heid, E.; Green, W.H. "Machine Learning of Reaction Properties via Learned
        Representations of the Condensed Graph of Reaction." J. Chem. Inf. Model. 2022, 62,
        2101-2110. https://doi.org/10.1021/acs.jcim.1c00975
    """

    atom_transform: TensorTransform[Atom]
    bond_transform: TensorTransform[Bond]
    mode_: InitVar[str | RxnMode] = RxnMode.REAC_DIFF

    def __post_init__(self, mode_: str | RxnMode):
        super().__post_init__()

        self.mode = mode_
        self.node_dim += len(self.atom_transform) - self.atom_transform.max_atomic_num - 1
        self.edge_dim *= 2

    @property
    def mode(self) -> RxnMode:
        return self.__mode

    @mode.setter
    def mode(self, m: str | RxnMode):
        self.__mode = RxnMode.get(m)

    def __call__(self, rxn: Rxn) -> Graph:
        reac, pdt = rxn
        r2p_idx_map, pdt_idxs, reac_idxs = self.map_reac_to_prod(reac, pdt)

        V = self._calc_node_feature_matrix(reac, pdt, r2p_idx_map, pdt_idxs, reac_idxs)
        E = []
        edge_index = [[], []]

        n_atoms_reac = reac.GetNumAtoms()

        i = 0
        for u in range(len(V)):
            for v in range(u + 1, len(V)):
                b_reac, b_prod = self._get_bonds(
                    reac, pdt, r2p_idx_map, pdt_idxs, n_atoms_reac, u, v
                )
                if b_reac is None and b_prod is None:
                    continue

                e = self._calc_edge_feature(b_reac, b_prod)
                E.extend([e, e])
                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])

                i += 2

        E = np.array(E)
        rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return Graph(V, E, edge_index, rev_edge_index)

    def _calc_node_feature_matrix(
        self,
        rct: Mol,
        pdt: Mol,
        r2p_idx_map: dict[int, int],
        pdt_idxs: Iterable[int],
        reac_idxs: Iterable[int],
    ) -> np.ndarray:
        """Calculate the node feature matrix for the reaction"""
        X_v_r1 = np.array([self.atom_transform(a) for a in rct.GetAtoms()])
        X_v_p2 = np.array([self.atom_transform(pdt.GetAtomWithIdx(i)) for i in pdt_idxs])
        X_v_p2 = X_v_p2.reshape(-1, X_v_r1.shape[1])

        if self.mode in [RxnMode.REAC_DIFF, RxnMode.PROD_DIFF, RxnMode.REAC_PROD]:
            # Reactant:
            # (1) regular features for each atom in the reactants
            # (2) zero features for each atom that's only in the products
            X_v_r2 = [self.atom_transform.num_only(pdt.GetAtomWithIdx(i)) for i in pdt_idxs]
            X_v_r2 = np.array(X_v_r2).reshape(-1, X_v_r1.shape[1])

            # Product:
            # (1) either (a) product-side features for each atom in both
            #         or (b) zero features for each atom only in the reatants
            # (2) regular features for each atom only in the products
            X_v_p1 = np.array(
                [
                    (
                        self.atom_transform(pdt.GetAtomWithIdx(r2p_idx_map[i]))
                        if (i := a.GetIdx()) not in reac_idxs
                        else self.atom_transform.num_only(a)
                    )
                    for a in rct.GetAtoms()
                ]
            )
        else:
            # Reactant:
            # (1) regular features for each atom in the reactants
            # (2) regular features for each atom only in the products
            X_v_r2 = [self.atom_transform(pdt.GetAtomWithIdx(i)) for i in pdt_idxs]
            X_v_r2 = np.array(X_v_r2).reshape(-1, X_v_r1.shape[1])

            # Product:
            # (1) either (a) product-side features for each atom in both
            #         or (b) reactant-side features for each atom only in the reatants
            # (2) regular features for each atom only in the products
            X_v_p1 = np.array(
                [
                    (
                        self.atom_transform(pdt.GetAtomWithIdx(r2p_idx_map[a.GetIdx()]))
                        if a.GetIdx() not in reac_idxs
                        else self.atom_transform(a)
                    )
                    for a in rct.GetAtoms()
                ]
            )

        X_v_r = np.concatenate((X_v_r1, X_v_r2))
        X_v_p = np.concatenate((X_v_p1, X_v_p2))

        m = min(len(X_v_r), len(X_v_p))

        if self.mode in [RxnMode.REAC_PROD, RxnMode.REAC_PROD_BALANCE]:
            X_v = np.hstack((X_v_r[:m], X_v_p[:m, self.atom_transform.max_atomic_num + 1 :]))
        else:
            X_v_d = X_v_p[:m] - X_v_r[:m]
            if self.mode in [RxnMode.REAC_DIFF, RxnMode.REAC_DIFF_BALANCE]:
                X_v = np.hstack((X_v_r[:m], X_v_d[:m, self.atom_transform.max_atomic_num + 1 :]))
            else:
                X_v = np.hstack((X_v_p[:m], X_v_d[:m, self.atom_transform.max_atomic_num + 1 :]))

        return X_v

    def _get_bonds(
        self,
        rct: Bond,
        pdt: Bond,
        ri2pj: Mapping[int, int],
        pids: Sequence[int],
        n_atoms_r: int,
        u: int,
        v: int,
    ) -> tuple[Bond | None, Bond | None]:
        """get the corresponding reactant- and product-side bond, respectively, betweeen atoms
        :attr:`u` and :attr:`v`"""
        if u >= n_atoms_r and v >= n_atoms_r:
            b_prod = pdt.GetBondBetweenAtoms(pids[u - n_atoms_r], pids[v - n_atoms_r])

            if self.mode in [
                RxnMode.REAC_PROD_BALANCE,
                RxnMode.REAC_DIFF_BALANCE,
                RxnMode.PROD_DIFF_BALANCE,
            ]:
                b_reac = b_prod
            else:
                b_reac = None
        elif u < n_atoms_r and v >= n_atoms_r:  # One atom only in product
            b_reac = None

            if u in ri2pj:
                b_prod = pdt.GetBondBetweenAtoms(ri2pj[u], pids[v - n_atoms_r])
            else:  # Atom atom only in reactant, the other only in product
                b_prod = None
        else:
            b_reac = rct.GetBondBetweenAtoms(u, v)

            if u in ri2pj and v in ri2pj:  # Both atoms in both reactant and product
                b_prod = pdt.GetBondBetweenAtoms(ri2pj[u], ri2pj[v])
            elif self.mode in [
                RxnMode.REAC_PROD_BALANCE,
                RxnMode.REAC_DIFF_BALANCE,
                RxnMode.PROD_DIFF_BALANCE,
            ]:
                b_prod = None if (u in ri2pj or v in ri2pj) else b_reac
            else:  # One or both atoms only in reactant
                b_prod = None

        return b_reac, b_prod

    def _calc_edge_feature(self, b_reac: Bond, b_pdt: Bond):
        """Calculate the global features of the two bonds"""
        x_e_r = self.bond_transform(b_reac)
        x_e_p = self.bond_transform(b_pdt)
        x_e_d = x_e_p - x_e_r

        if self.mode in [RxnMode.REAC_PROD, RxnMode.REAC_PROD_BALANCE]:
            x_e = np.hstack((x_e_r, x_e_p))
        elif self.mode in [RxnMode.REAC_DIFF, RxnMode.REAC_DIFF_BALANCE]:
            x_e = np.hstack((x_e_r, x_e_d))
        else:
            x_e = np.hstack((x_e_p, x_e_d))

        return x_e

    @classmethod
    def map_reac_to_prod(cls, reacs: Mol, pdts: Mol) -> tuple[dict[int, int], list[int], list[int]]:
        """Map atom indices between corresponding atoms in the reactant and product molecules

        Parameters
        ----------
        reacs : Mol
            An RDKit molecule of the reactants
        pdts : Mol
            An RDKit molecule of the products

        Returns
        -------
        dict[int, int]
            A dictionary of corresponding atom indices from reactant atoms to product atoms
        list[int]
            atom indices of product atoms
        list[int]
            atom indices of reactant atoms
        """
        pdt_idxs = []
        mapno2pj = {}
        reac_atommap_nums = {a.GetAtomMapNum() for a in reacs.GetAtoms()}

        for atom in pdts.GetAtoms():
            map_num = atom.GetAtomMapNum()
            j = atom.GetIdx()

            if map_num > 0:
                mapno2pj[map_num] = j
                if map_num not in reac_atommap_nums:
                    pdt_idxs.append(j)
            else:
                pdt_idxs.append(j)

        rct_idxs = []
        r2p_idx_map = {}

        for atom in reacs.GetAtoms():
            map_num = atom.GetAtomMapNum()
            i = atom.GetIdx()

            if map_num > 0:
                try:
                    r2p_idx_map[i] = mapno2pj[map_num]
                except KeyError:
                    rct_idxs.append(i)
            else:
                rct_idxs.append(i)

        return r2p_idx_map, pdt_idxs, rct_idxs


CGRFeaturizer = CondensedReactionGraphFeaturizer

raise ImportError
