from dataclasses import dataclass, field

import numpy as np
import torch

from mol_gnn.data.models.graph import Graph
from mol_gnn.transforms.atom import AtomTransform, MultiTypeAtomTransform
from mol_gnn.transforms.base import Transform
from mol_gnn.transforms.bond import BondTransform, MultiTypeBondTransform
from mol_gnn.types import Mol


@dataclass
class MolToGraph(Transform[Mol, Graph]):
    atom_transform: AtomTransform = field(default_factory=MultiTypeAtomTransform)
    bond_transform: BondTransform = field(default_factory=MultiTypeBondTransform)

    @property
    def node_dim(self) -> int:
        return len(self.atom_transform)

    @property
    def edge_dim(self) -> int:
        return len(self.bond_transform)

    def __call__(self, mol: Mol) -> Graph:
        V = self.atom_transform(mol.GetAtoms())
        E = self.bond_transform(mol.GetBonds())

        edge_index = [
            [(u := bond.GetBeginAtomIdx(), v := bond.GetEndAtomIdx()), (v, u)]
            for bond in mol.GetBonds()
        ]
        edge_index = torch.tensor(sum(edge_index, start=[])).T
        rev_index = torch.from_numpy(np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel())

        return Graph(V, E, edge_index, rev_index)
