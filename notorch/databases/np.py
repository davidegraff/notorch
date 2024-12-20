from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from os import PathLike
from typing import Final

import numpy as np
from numpy.lib.npyio import NpzFile
from torch import Tensor

from notorch.databases.base import Database
from notorch.utils.mixins import CollateNDArrayMixin


@dataclass
class NPZDatabase(CollateNDArrayMixin, Database[int, np.ndarray, Tensor]):
    path: Final[PathLike]
    key: Final[str]

    npz: NpzFile | None = field(init=False, default=None, repr=False)
    X: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        with np.load(self.path) as npz:
            self.X = npz[self.key]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.X.shape

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.X)


@dataclass
class NPYDatabase(NPZDatabase):
    path: Final[PathLike]
    mmap_mode: InitVar[str | None] = None

    def __post_init__(self, mmap_mode: str | None):
        # with np.load(self.path) as npz:
        #     self.shape = npz[self.key].shape

        self.npz = None
        self.X = np.load(self.path, mmap_mode)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.X.shape

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.X)

    # def __getitem__(self, idx: int) -> np.ndarray:
    #     try:
    #         return self.X[idx]
    #     except TypeError as e:
    #         raise ClosedDatabaseError(type(self)) from e

    # def __enter__(self) -> Self:
    #     self.npz: NpzFile = np.load(self.path, "r")
    #     self.X = self.npz[self.key]

    #     return self

    # def __exit__(self, *exc):
    #     if self.npz is not None:
    #         self.npz.close()

    #     self.npz = None
    #     self.X = None

    # def __iter__(self) -> Iterator[int]:
    #     with self:
    #         return iter(range(len(self)))


if __name__ == "__main__":
    db = NPZDatabase("foo.npz", "x")
    db[1]
