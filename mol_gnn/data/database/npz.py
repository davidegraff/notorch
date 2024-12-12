from collections.abc import Iterator
from dataclasses import dataclass, field
from os import PathLike
from typing import Final, Self

import numpy as np
from numpy.lib.npyio import NpzFile

from mol_gnn.data.database.base import Database
from mol_gnn.exceptions import ClosedDatabaseError


@dataclass
class NPZDatabase(Database[int, np.ndarray]):
    path: Final[PathLike]
    key: Final[str]

    shape: Final[tuple[int, ...]] = field(init=False, repr=True)
    npz: NpzFile | None = field(init=False, default=None, repr=False)
    X: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        with np.load(self.path) as npz:
            self.shape = npz[self.key].shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        try:
            return self.X[idx]
        except TypeError as e:
            raise ClosedDatabaseError(type(self)) from e

    def __enter__(self) -> Self:
        self.npz: NpzFile = np.load(self.path, "r")
        self.X = self.npz[self.key]

        return self

    def __exit__(self, *exc):
        if self.npz is not None:
            self.npz.close()

        self.npz = None
        self.X = None

    def __iter__(self) -> Iterator[int]:
        with self:
            return iter(range(len(self)))


if __name__ == "__main__":
    db = NPZDatabase("foo.npz", "x")
    db[1]
