from dataclasses import dataclass, field
from typing import Iterable

from jaxtyping import Int
import numpy as np
import torch
from torch import Tensor

from mol_gnn.data.models.graph import Graph
from mol_gnn.transforms.atom import MultiTypeAtomTransform
from mol_gnn.transforms.base import Transform
from mol_gnn.transforms.bond import MultiTypeBondTransform
from mol_gnn.types import Atom, Bond, Mol


@dataclass
class MolToGraph(Transform[Mol, Graph]):
    atom_transform: Transform[Iterable[Atom], Int[Tensor, "V t_v"]] = field(
        default_factory=MultiTypeAtomTransform
    )
    bond_transform: Transform[Iterable[Bond], Int[Tensor, "E t_e"]] = field(
        default_factory=MultiTypeBondTransform
    )

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
