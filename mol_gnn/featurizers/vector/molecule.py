import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from mol_gnn.featurizers.vector.base import VectorFeaturizer
from mol_gnn.utils import ClassRegistry
from mol_gnn.types import Mol


MoleculeFeaturizerRegistry = ClassRegistry[VectorFeaturizer[Mol]]()


class MorganFeaturizerMixin:
    def __init__(self, radius: int = 2, length: int = 2048, include_chirality: bool = True):
        if radius < 0:
            raise ValueError(f"arg 'radius' must be >= 0! got: {radius}")

        self.length = length
        self.F = GetMorganGenerator(
            radius=radius, fpSize=length, includeChirality=include_chirality
        )

    def __len__(self) -> int:
        return self.length


class BinaryFeaturizerMixin:
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return self.F.GetFingerprintAsNumPy(mol)


class CountFeaturizerMixin:
    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        return self.F.GetCountFingerprintAsNumPy(mol)


@MoleculeFeaturizerRegistry("morgan_binary")
class MorganBinaryFeaturzer(MorganFeaturizerMixin, BinaryFeaturizerMixin, VectorFeaturizer[Mol]):
    pass


@MoleculeFeaturizerRegistry("morgan_count")
class MorganCountFeaturizer(MorganFeaturizerMixin, CountFeaturizerMixin, VectorFeaturizer[Mol]):
    pass
