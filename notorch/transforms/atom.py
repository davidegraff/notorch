from __future__ import annotations

from collections.abc import Iterable, Sequence
import textwrap
from typing import Protocol

from jaxtyping import Int
from rdkit.Chem.rdchem import Atom, ChiralType, HybridizationType
import torch
from torch import Tensor
from torch.nn import functional as F

from notorch.conf import REPR_INDENT
from notorch.transforms.conf import (
    CHIRAL_TAGS,
    DEGREES,
    ELEMENTS,
    FORMAL_CHARGES,
    HYBRIDIZATIONS,
    NUM_HS,
)
from notorch.transforms.utils.inverse_index import InverseIndexWithUnknown, build


class AtomTransform(Protocol):
    def __len__(self) -> int: ...
    def __call__(self, input: Iterable[Atom]) -> Int[Tensor, "n t"]: ...


class ElementOnlyAtomTransform:
    def __init__(self, elements: Sequence[str] = ELEMENTS):
        self.element_map = InverseIndexWithUnknown(elements)

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

        self.__num_types = sum(sizes)
        self.sizes = torch.tensor(sizes)
        self.offset = F.pad(self.sizes.cumsum(dim=0), [1, 0])[:-1]

    def __len__(self) -> int:
        return self.__num_types

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

    def __repr__(self) -> str:
        lines = []

        if self.element_map is not None:
            lines.append(f"(elements): {self.element_map}")
        if self.hybrid_map is not None:
            lines.append(f"(hybridizations): {self.hybrid_map}")
        if self.chirality_map is not None:
            lines.append(f"(chiralities): {self.chirality_map}")
        if self.degree_map is not None:
            lines.append(f"(degrees): {self.degree_map}")
        if self.fc_map is not None:
            lines.append(f"(formal_charges): {self.fc_map}")
        if self.num_hs_map is not None:
            lines.append(f"(num_hs): {self.num_hs_map}")
        if self.aromaticity_map is not None:
            lines.append(f"(aromaticity): {self.aromaticity_map}")
        text = "\n".join(lines)

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])
