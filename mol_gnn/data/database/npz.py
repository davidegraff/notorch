from dataclasses import dataclass, field
from os import PathLike

import numpy as np
from numpy.lib.npyio import NpzFile


@dataclass
class NPZDatabase:
    path: PathLike
    key: str

    npz: NpzFile | None = field(init=False, default=None)
    X: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        with open(self.path) as npz:
            self.__size = len(npz[self.key])

    def __len__(self) -> int:
        return self.__size

    def __getitem__(self, idx: int):
        try:
            return self.X[idx]
        except TypeError as e:
            raise ValueError from e

    def __enter__(self):
        self.npz: NpzFile = np.load(self.path, "r")
        self.X = self.npz[self.key]

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.npz is not None:
            self.npz.close()

        self.npz = None
        self.X = None
