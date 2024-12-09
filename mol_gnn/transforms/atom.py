from __future__ import annotations

from collections.abc import Iterable, Sequence, Sized
from typing import Protocol

from jaxtyping import Int
from rdkit.Chem.rdchem import ChiralType, HybridizationType
import torch
from torch import Tensor
from torch.nn import functional as F

from mol_gnn.types import Atom
from mol_gnn.transforms.utils.index_map import IndexMapWithUnknown, build

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3]
HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
]
NUM_HS = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-1, -2, 1, 2, 0]


class AtomTransform(Protocol):
    def __len__(self) -> int: ...

    def __call__(self, input: Iterable[Atom]) -> Int[Tensor, "n t"]: ...


class ElementOnlyAtomTransform:
    def __init__(self, elements: Sequence[str] = ELEMENTS):
        self.element_map = IndexMapWithUnknown(elements)

    def __len__(self) -> int:
        return len(self.element_map)

    @property
    def num_types(self) -> int:
        return 1

    def __call__(self, input: Iterable[Atom]) -> Int[Tensor, "n 1"]:
        types = [[self.element_map[atom.GetSymbol()]] for atom in input]

        return torch.tensor(types)


class MultiTypeAtomTransform:
    def __init__(
        self,
        elements: Sequence[str] | None = ELEMENTS,
        hybridizations: Sequence[HybridizationType] | None = HYBRIDIZATIONS,
        chiral_tags: Sequence[ChiralType] | None = CHIRAL_TAGS,
        degrees: Sequence[int] | None = DEGREES,
        formal_charges: Sequence[int] | None = FORMAL_CHARGES,
        num_hs: Sequence[int] | None = NUM_HS,
        include_aromaticity: bool = True,
    ):
        aromaticity_choices = [True, False] if include_aromaticity else None

        self.element_map = build(elements, unknown_pad=True)
        self.hybrid_map = build(hybridizations, unknown_pad=True)
        self.chirality_map = build(chiral_tags, unknown_pad=True)
        self.degree_map = build(degrees, unknown_pad=True)
        self.fc_map = build(formal_charges, unknown_pad=True)
        self.num_hs_map = build(num_hs, unknown_pad=True)
        self.aromaticity_map = build(aromaticity_choices, unknown_pad=False)

        sizes = [
            len(index_map)
            for index_map in [
                self.element_map,
                self.hybrid_map,
                self.chirality_map,
                self.degree_map,
                self.fc_map,
                self.num_hs_map,
                self.aromaticity_map,
            ]
            if index_map is not None
        ]

        self.sizes = torch.tensor(sizes)
        self.offset = F.pad(self.sizes.cumsum(dim=0), [1, 0])[:-1]

    def __len__(self) -> int:
        return self.sizes.sum()

    @property
    def num_types(self) -> int:
        return len(self.sizes)

    def _transform_single(self, atom: Atom) -> list[int]:
        types = []

        if self.element_map is not None:
            types.append(self.element_map[atom.GetSymbol()])
        if self.hybrid_map is not None:
            types.append(self.hybrid_map[atom.GetHybridization()])
        if self.chirality_map is not None:
            types.append(self.chirality_map[atom.GetChiralTag()])
        if self.degree_map is not None:
            types.append(self.degree_map[atom.GetTotalDegree()])
        if self.fc_map is not None:
            types.append(self.fc_map[atom.GetFormalCharge()])
        if self.num_hs_map is not None:
            types.append(self.num_hs_map[atom.GetTotalNumHs()])
        if self.aromaticity_map is not None:
            types.append(self.aromaticity_map[atom.GetIsAromatic()])

        return types

    def __call__(self, input: Iterable[Atom]) -> Int[Tensor, "n t"]:
        types = [self._transform_single(atom) for atom in input]

        return torch.tensor(types) + self.offset.unsqueeze(0)
