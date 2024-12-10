from collections.abc import Callable, Collection, Sized
from dataclasses import InitVar, dataclass

from jaxtyping import Float
from numpy.typing import NDArray
from rdkit.Chem.rdFingerprintGenerator import FingeprintGenerator64, GetMorganGenerator
import torch
from torch import Tensor

from mol_gnn.transforms.base import Transform
from mol_gnn.types import Mol


@dataclass
class MolToFP(Sized, Transform[Mol, Float[Tensor, "d"], Float[Tensor, "n d"]]):
    fpgen: FingeprintGenerator64
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

    def collate(self, inputs: Collection[Tensor]) -> Tensor:
        return torch.stack(inputs)

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
