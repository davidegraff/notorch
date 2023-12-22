from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from mol_gnn.featurizers.molgraph.molgraph import MolGraph

T = TypeVar("T")


class MolGraphFeaturizer(Callable, Generic[T]):
    """A :class:`MolGraphFeaturizer` featurizes inputs into :class:`MolGraph`s"""

    @abstractmethod
    def __call__(self, x: T) -> MolGraph:
        """Featurize the input :attr:`x` into a :class:`MolGraph`

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
