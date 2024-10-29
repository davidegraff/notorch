from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from functools import singledispatchmethod

from jaxtyping import Float
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import FingeprintGenerator64, GetMorganGenerator
import torch
from torch import Tensor

from mol_gnn.transforms.base import TensorTransform
from mol_gnn.types import Mol


@dataclass
class FingerprintFeaturizer(TensorTransform[Mol]):
    fpgen: FingeprintGenerator64
    bit_fingerprint: InitVar[bool] = True

    def __post_init__(self, bit_fingerprint: bool = True):
        self.func = (
            self.fpgen.GetFingerprintAsNumPy
            if bit_fingerprint
            else self.fpgen.GetCountFingerprintAsNumPy
        )

    def __len__(self) -> int:
        return self.fpgen.GetOptions().fpSize

    @singledispatchmethod
    def __call__(self, input) -> Float[Tensor, "*n d"]:
        fp = self.func(input)

        return torch.from_numpy(fp).float()

    @__call__.register
    def _(self, input: Iterable[Mol]):
        fps = [self.func(mol) for mol in input]

        return torch.from_numpy(np.stack(fps)).float()

    @classmethod
    def morgan(
        cls,
        radius: int = 2,
        length: int = 2048,
        include_chirality: bool = True,
        bit_fingerprint: bool = True,
    ):
        fpgen = GetMorganGenerator(radius=radius, fpSize=length, includeChirality=include_chirality)

        return cls(fpgen, bit_fingerprint)
