from dataclasses import dataclass, field
import textwrap
from typing import ClassVar

import numpy as np
import torch

from notorch.conf import REPR_INDENT
from notorch.data.models.graph import BatchedGraph, Graph
from notorch.transforms.atom import AtomTransform, MultiTypeAtomTransform
from notorch.transforms.base import Transform
from notorch.transforms.bond import BondTransform, MultiTypeBondTransform
from notorch.types import Mol


@dataclass(repr=False)
class MolToGraph(Transform[Mol, Graph, BatchedGraph]):
    _in_key_: ClassVar[str] = "mol"
    _out_key_: ClassVar[str] = "G"

    atom_transform: AtomTransform = field(default_factory=MultiTypeAtomTransform)
    bond_transform: BondTransform = field(default_factory=MultiTypeBondTransform)

    @property
    def num_node_types(self) -> int:
        return len(self.atom_transform)

    @property
    def num_edge_types(self) -> int:
        return len(self.bond_transform)

    def __call__(self, mol: Mol) -> Graph:
        V = self.atom_transform(mol.GetAtoms())
        E = self.bond_transform(mol.GetBonds()).repeat_interleave(2, dim=0)

        edge_index = [
            [(u := bond.GetBeginAtomIdx(), v := bond.GetEndAtomIdx()), (v, u)]
            for bond in mol.GetBonds()
        ]
        edge_index = torch.tensor(sum(edge_index, start=[])).T
        rev_index = torch.from_numpy(np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel())

        return Graph(V, E, edge_index, rev_index)

    collate = BatchedGraph.from_graphs

    def __repr__(self) -> str:
        text = "\n".join(
            [f"(atom_transform): {self.atom_transform}", f"(bond_transform): {self.bond_transform}"]
        )

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])
