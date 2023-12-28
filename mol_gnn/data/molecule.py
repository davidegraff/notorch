from dataclasses import dataclass, field
from typing import Self

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from mol_gnn.types import Mol
from mol_gnn.data.mixins import _DatapointMixin, _MolGraphDatasetMixin, Datum
from mol_gnn.featurizers import VectorFeaturizer, GraphFeaturizer, MolGraphFeaturizer
from mol_gnn.utils.chem import make_mol


@dataclass
class _MoleculeDatapointMixin:
    mol: Chem.Mol
    """the molecule associated with this datapoint"""

    @classmethod
    def from_smi(
        cls, smi: str, *args, keep_h: bool = False, add_h: bool = False, **kwargs
    ) -> Self:
        mol = make_mol(smi, keep_h, add_h)

        return cls(mol, *args, **kwargs)


@dataclass
class MoleculeDatapoint(_DatapointMixin, _MoleculeDatapointMixin):
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and
    targets."""

    V_f: np.ndarray | None = None
    """a numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
    ``d_vf`` is the number of additional features that will be concatenated to atom-level features
    *before* message passing"""
    E_f: np.ndarray | None = None
    """A numpy array of shape ``E x d_ef``, where ``E`` is the number of bonds in the molecule, and
    ``d_ef`` is the number of additional features  containing additional features that will be
    concatenated to bond-level features *before* message passing"""
    V_d: np.ndarray | None = None
    """A numpy array of shape ``V x d_vd``, where ``V`` is the number of atoms in the molecule, and
    ``d_vd`` is the number of additional features that will be concatenated to atom-level features
    *after* message passing"""

    def __post_init__(self, mfs: list[VectorFeaturizer[Mol]] | None):
        if self.mol is None:
            raise ValueError("Input molecule was `None`!")

        NAN_TOKEN = 0

        if self.V_f is not None:
            self.V_f[np.isnan(self.V_f)] = NAN_TOKEN
        if self.E_f is not None:
            self.E_f[np.isnan(self.E_f)] = NAN_TOKEN
        if self.V_d is not None:
            self.V_d[np.isnan(self.V_d)] = NAN_TOKEN

        super().__post_init__(mfs)

    def calc_features(self, mfs: list[VectorFeaturizer[Mol]]) -> np.ndarray:
        if self.mol.GetNumHeavyAtoms() == 0:
            return np.zeros(sum(len(mf) for mf in mfs))

        return np.hstack([mf(self.mol) for mf in mfs])


@dataclass
class MoleculeDataset(_MolGraphDatasetMixin, Dataset):
    """A :class:`~torch.utils.data.Datset` composed of :class:`MoleculeDatapoint`s

    Parameters
    ----------
    data : Iterable[MoleculeDatapoint]
        the data from which to create a dataset
    featurizer : Featurizer[Mol]
        the featurizer with which to generate MolGraphs of the molecules
    """

    data: list[MoleculeDatapoint]
    featurizer: GraphFeaturizer[Mol] = field(default_factory=MolGraphFeaturizer)

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.featurizer(d.mol)#self.V_fs[idx], self.E_fs[idx])

        return Datum(mg, self.V_ds[idx], self.X_f[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask)

    @property
    def smiles(self) -> list[str]:
        """the SMILES strings associated with the dataset"""
        return [Chem.MolToSmiles(d.mol) for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        """the molecules associated with the dataset"""
        return [d.mol for d in self.data]

    @property
    def _V_fs(self) -> np.ndarray:
        """the raw atom features of the dataset"""
        return np.array([d.V_f for d in self.data])

    @property
    def V_fs(self) -> np.ndarray:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_fs

    @V_fs.setter
    def V_fs(self, V_fs: np.ndarray):
        """the (scaled) atom features of the dataset"""
        self._validate_attribute(V_fs, "atom features")

        self.__V_fs = np.array(V_fs)

    @property
    def _E_fs(self) -> np.ndarray:
        """the raw bond features of the dataset"""
        return np.array([d.E_f for d in self.data])

    @property
    def E_fs(self) -> np.ndarray:
        """the (scaled) bond features of the dataset"""
        return self.__E_fs

    @E_fs.setter
    def E_fs(self, E_fs: np.ndarray):
        self._validate_attribute(E_fs, "bond features")

        self.__E_fs = np.array(E_fs)

    @property
    def _V_ds(self) -> np.ndarray:
        """the raw atom descriptors of the dataset"""
        return np.array([d.V_d for d in self.data])

    @property
    def V_ds(self) -> np.ndarray:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_ds

    @V_ds.setter
    def V_ds(self, V_ds: np.ndarray):
        self._validate_attribute(V_ds, "atom descriptors")

        self.__V_ds = np.array(V_ds)

    @property
    def d_vf(self) -> int:
        """the extra atom feature dimension, if any"""
        return 0 if np.equal(self.V_fs, None).all() else self.V_fs[0].shape[1]

    @property
    def d_ef(self) -> int:
        """the extra bond feature dimension, if any"""
        return 0 if np.equal(self.E_fs, None).all() else self.E_fs[0].shape[1]

    @property
    def d_vd(self) -> int:
        """the extra atom descriptor dimension, if any"""
        return 0 if np.equal(self.V_ds, None).all() else self.V_ds[0].shape[1]

    def normalize_inputs(
        self, key: str | None = "X_f", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_f", "V_f", "E_f", "V_d", None}

        match key:
            case "X_f":
                X = self.X_f
            case "V_f":
                X = None if self.V_fs is None else np.concatenate(self.V_fs, axis=0)
            case "E_f":
                X = None if self.E_fs is None else np.concatenate(self.E_fs, axis=0)
            case "V_d":
                X = None if self.V_ds is None else np.concatenate(self.V_ds, axis=0)
            case None:
                return [self.normalize_inputs(k, scaler) for k in VALID_KEYS - {None}]
            case _:
                ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        match key:
            case "X_f":
                self.X_f = scaler.transform(X)
            case "V_f":
                self.V_fs = [scaler.transform(V_f) for V_f in self.V_fs]
            case "E_f":
                self.E_fs = [scaler.transform(E_f) for E_f in self.E_fs]
            case "V_d":
                self.V_ds = [scaler.transform(V_d) for V_d in self.V_ds]
            case _:
                raise RuntimeError("unreachable code reached!")

        return scaler

    def reset(self):
        """reset the {atom, bond, molecule} features and targets of each datapoint to its raw
        value"""
        super().reset()

        self.__V_fs = self._V_fs
        self.__E_fs = self._E_fs
        self.__V_ds = self._V_ds
