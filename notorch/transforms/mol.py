from collections.abc import Callable, Sized
from dataclasses import InitVar, dataclass
from typing import ClassVar

from jaxtyping import Float
from numpy.typing import NDArray
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64, GetMorganGenerator
import torch
from torch import Tensor

from notorch.transforms.base import Transform
from notorch.types import Mol
from notorch.utils.mixins import CollateNDArrayMixin


@dataclass
class MolToFP(
    CollateNDArrayMixin, Sized, Transform[Mol, Float[NDArray, "d"], Float[Tensor, "n d"]]
):
    _in_key_: ClassVar[str] = "mol"
    _out_key_: ClassVar[str] = "fp"

    fpgen: FingerprintGenerator64
    bit_fingerprint: InitVar[bool] = True

    def __post_init__(self, bit_fingerprint: bool = True):
        self.func: Callable[[Mol], Float[NDArray, "d"]] = (
            self.fpgen.GetFingerprintAsNumPy
            if bit_fingerprint
            else self.fpgen.GetCountFingerprintAsNumPy
        )

    def __len__(self) -> int:
        return self.fpgen.GetOptions().fpSize

    def __call__(self, input: Mol) -> Float[Tensor, "d"]:
        fp = self.func(input)

        return torch.from_numpy(fp).float()

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
