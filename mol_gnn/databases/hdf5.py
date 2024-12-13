from collections.abc import Iterator
from dataclasses import dataclass, field
from os import PathLike
from typing import Final, Self

import h5py
import numpy as np

from mol_gnn.databases.base import Database
from mol_gnn.exceptions import ClosedDatabaseError


@dataclass
class HDF5Database(Database[int, np.ndarray]):
    path: Final[PathLike]
    dataset: Final[str]

    def __post_init__(self):
        with h5py.File(self.path) as h5f:
            self.X = h5f[self.dataset][:]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.X.shape

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx]

    def __iter__(self) -> Iterator[int]:
        return iter(self.X)


@dataclass
class __HDF5DatabaseOnDisk(HDF5Database):
    path: Final[PathLike]
    dataset: Final[str]

    shape: Final[tuple[int, ...]] = field(init=False, repr=True)
    h5f: h5py.File | None = field(init=False, default=None, repr=False)
    dataset: h5py.Dataset | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        with h5py.File(self.path) as h5f:
            self.shape = h5f[self.dataset].shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        try:
            return self.dataset[idx]
        except TypeError as e:
            raise ClosedDatabaseError(type(self)) from e

    def __enter__(self) -> Self:
        self.h5f = h5py.File(self.path, "r")
        self.dataset = self.h5f[self.dataset]

        return self

    def __exit__(self, *exc):
        if self.h5f is not None:
            self.h5f.close()

        self.h5f = None
        self.dataset = None

    def __iter__(self) -> Iterator[int]:
        with self:
            return iter(range(len(self)))
