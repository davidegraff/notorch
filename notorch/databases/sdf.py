from dataclasses import dataclass, field
from os import PathLike
from typing import Iterator, Self

import rdkit.Chem as Chem

from notorch.databases.base import Database
from notorch.types import Mol


@dataclass
class SDFDatabase(Database[int, Mol, list[Mol]]):
    path: PathLike

    mols: list[Mol] = field(init=False)

    def __post_init__(self):
        with open(self.path) as supp:
            self.mols = list(supp)

    def __len__(self) -> int:
        return len(self.mols)

    def __getitem__(self, idx: int) -> Mol:
        return self.mols[idx]

    def __iter__(self) -> Iterator[Mol]:
        return iter(self.mols)

    collate = list


@dataclass
class __SDFDatabaseOnDisk(SDFDatabase):
    path: PathLike

    def __post_init__(self):
        with open(self.path) as f:
            self.__size = sum(1 for line in f if line.startswith("$$$$"))
        self.supp = None

    def __len__(self) -> int:
        return self.__size

    def __getitem__(self, idx: int) -> Mol:
        try:
            return self.supp[idx]
        except TypeError:
            raise ValueError

    def __enter__(self) -> Self:
        self.supp = Chem.SDMolSupplier(str(self.path))

        return self

    def __exit__(self, *exc):
        self.supp = None

    def __iter__(self) -> Iterator:
        with self:
            return iter(self.supp)
