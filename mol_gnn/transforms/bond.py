from __future__ import annotations

from collections.abc import Iterable, Sized

from jaxtyping import Int
from rdkit.Chem import Bond
from rdkit.Chem.rdchem import BondStereo, BondType
import torch
from torch import Tensor
from torch.nn import functional as F

from mol_gnn.transforms.base import Transform
from mol_gnn.transforms.utils.index_map import IndexMapWithUnknown, build

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


class TypeOnlyTransform(Sized, Transform[Iterable[Bond], Int[Tensor, "E 1"]]):
    def __init__(self, bond_types: Iterable[BondType] = BOND_TYPES):
        self.bond_type_map = IndexMapWithUnknown(bond_types)

    def __len__(self) -> int:
        return len(self.bond_type_map)

    def __call__(self, input: Iterable[Bond]) -> Int[Tensor, "*n d"]:
        types = [[self.bond_type_map[bond.GetBondType()]] for bond in input]

        return torch.tensor(types)

    # @singledispatchmethod
    # def __call__(self, input: Bond) -> Int[Tensor, "d"]:
    #     types = [self.bond_type_map[input.GetBondType()]]

    #     return torch.tensor(types)


class MultiTypeBondTransform(Sized, Transform[Iterable[Bond], Int[Tensor, "E t"]]):
    def __init__(
        self,
        bond_types: Iterable[BondType] | None = BOND_TYPES,
        stereos: Iterable[BondStereo] | None = BOND_STEREOS,
    ):
        self.bond_type_map = build(bond_types, unknown_pad=True)
        self.stereo_map = build(stereos, unknown_pad=True)

        sizes = [
            len(index_map)
            for index_map in [self.bond_type_map, self.stereo_map]
            if index_map is not None
        ]

        self.sizes = torch.tensor(sizes)
        self.offset = F.pad(self.sizes.cumsum(dim=0), [1, 0])[:-1]

    def __len__(self) -> int:
        return self.sizes.sum()

    def _transform_single(self, bond: Bond) -> list[int]:
        types = []

        if self.bond_type_map is not None:
            types.append(self.element_map[bond.GetBondType()])
        if self.stereo_map is not None:
            types.append(self.hybrid_map[bond.GetStereo()])

        return types

    def __call__(self, input: Iterable[Bond]) -> Int[Tensor, "E t"]:
        types = [self._transform_single(bond) for bond in input]

        return torch.tensor(types) + self.offset.unsqueeze(0)

    # @singledispatchmethod
    # def __call__(self, input: Bond) -> Int[Tensor, "d"]:
    #     types = self._transform_single(input)

    #     return torch.tensor(types) + self.offset
