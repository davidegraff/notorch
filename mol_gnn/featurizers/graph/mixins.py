from dataclasses import dataclass, field

from mol_gnn.types import Atom, Bond
from mol_gnn.featurizers.vector.base import VectorFeaturizer
from mol_gnn.featurizers.vector.atom import MultiHotAtomFeaturizer
from mol_gnn.featurizers.vector.bond import MultiHotBondFeaturizer


@dataclass(repr=False, eq=False)
class _MolGraphFeaturizerMixin:
    atom_featurizer: VectorFeaturizer[Atom] = field(default_factory=MultiHotAtomFeaturizer)
    bond_featurizer: VectorFeaturizer[Bond] = field(default_factory=MultiHotBondFeaturizer)

    def __post_init__(self):
        self.node_dim = len(self.atom_featurizer)
        self.edge_dim = len(self.bond_featurizer)

    @property
    def shape(self) -> tuple[int, int]:
        return self.node_dim, self.edge_dim
