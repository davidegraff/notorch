from collections.abc import Sized
from dataclasses import InitVar, dataclass

from jaxtyping import Float
from rdkit.Chem.rdFingerprintGenerator import FingeprintGenerator64, GetMorganGenerator
import torch
from torch import Tensor

from mol_gnn.transforms.base import Transform
from mol_gnn.types import Mol


@dataclass
class MolToFP(Sized, Transform[Mol, Float[Tensor, "d"], Float[Tensor, "n d"]]):
    fpgen: FingeprintGenerator64
    bit_fingerprint: InitVar[bool] = True

    def __post_init__(self, use_count: bool = True):
        self.func = (
            self.fpgen.GetCountFingerprintAsNumPy if use_count else self.fpgen.GetFingerprintAsNumPy
        )

    def __len__(self) -> int:
        return self.fpgen.GetOptions().fpSize

    def __call__(self, input) -> Float[Tensor, "d"]:
        fp = self.func(input)

        return torch.from_numpy(fp).float()

    def collate(self, inputs: list[Tensor]) -> Tensor:
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
