from __future__ import annotations

from collections.abc import Collection, Iterable
import textwrap
from typing import Protocol

from jaxtyping import Int
from rdkit.Chem.rdchem import BondStereo, BondType
from torch import Tensor
import torch
import torch.nn.functional as F

from mol_gnn.conf import REPR_INDENT
from mol_gnn.transforms.utils.inverse_index import InverseIndexWithUnknown, build
from mol_gnn.types import Bond

BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_STEREOS = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
    BondStereo.STEREOATROPCW,
]


class BondTransform(Protocol):
    def __len__(self) -> int: ...
    def __call__(self, input: Iterable[Bond]) -> Int[Tensor, "n t"]: ...


class BondTypeOnlyTransform:
    def __init__(self, bond_types: Collection[BondType] = BOND_TYPES):
        self.bond_type_map = InverseIndexWithUnknown(bond_types)

    def __len__(self) -> int:
        return len(self.bond_type_map)

    def __call__(self, input: Iterable[Bond]) -> Int[Tensor, "n 1"]:
        types = [[self.bond_type_map[bond.GetBondType()]] for bond in input]

        return torch.tensor(types)

    def __repr__(self):
        text = f"(bond_types): {self.bond_type_map}"

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])


class MultiTypeBondTransform:
    def __init__(
        self,
        bond_types: Collection[BondType] | None = BOND_TYPES,
        stereos: Collection[BondStereo] | None = BOND_STEREOS,
    ):
        self.bond_type_map = build(bond_types, unknown_pad=True)
        self.stereo_map = build(stereos, unknown_pad=True)

        sizes = [
            len(index_map)
            for index_map in [self.bond_type_map, self.stereo_map]
            if index_map is not None
        ]

        self.__num_types = sum(sizes)
        self.sizes = torch.tensor(sizes, dtype=torch.long)
        self.offset = F.pad(self.sizes.cumsum(dim=0), [1, 0])[:-1]

    def __len__(self) -> int:
        return self.__num_types

    def _transform_single(self, bond: Bond) -> list[int]:
        types = []

        if self.bond_type_map is not None:
            types.append(self.bond_type_map[bond.GetBondType()])
        if self.stereo_map is not None:
            types.append(self.stereo_map[bond.GetStereo()])

        return types

    def __call__(self, input: Iterable[Bond]) -> Int[Tensor, "n t"]:
        types = [self._transform_single(bond) for bond in input]

        return torch.tensor(types) + self.offset.unsqueeze(0)

    def __repr__(self) -> str:
        lines = []

        if self.bond_type_map is not None:
            lines.append(f"(bond_types): {self.stereo_map}")
        if self.stereo_map is not None:
            lines.append(f"(stereos): {self.stereo_map}")
        text = "\n".join(lines)

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])

    def stringify_choices(self):
        return list(map(str, ))
