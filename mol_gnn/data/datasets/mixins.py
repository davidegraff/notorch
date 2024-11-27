from dataclasses import InitVar, dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler

from mol_gnn.types import Mol
from mol_gnn.featurizers.vector.base import VectorFeaturizer


@dataclass(slots=True)
class _DatapointMixin:
    """A mixin class for molecule-, reaction-, and multicomponent-type data"""

    y: np.ndarray | None = None
    """the target vector."""
    weight: float = 1
    """the sample weight."""
    gt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form :math:`< x`"""
    lt_mask: np.ndarray | None = None
    """Indicates whether the targets are an inequality regression target of the form :math:`> x`"""
    x_f: np.ndarray | None = None
    """A vector of length ``d_f`` containing additional features (e.g., Morgan fingerprint) to
    concatenate to the global representation *after* aggregation"""
    mfs: InitVar[list[VectorFeaturizer[Mol]] | None] = None
    """A list of molecule featurizers to use"""

    def __post_init__(self, mfs: list[VectorFeaturizer[Mol]] | None):
        if self.x_f is not None and mfs is not None:
            raise ValueError("Cannot provide both loaded features and molecular featurizers!")

        if mfs is not None:
            self.x_f = self.calc_features(mfs)

        NAN_TOKEN = 0
        if self.x_f is not None:
            self.x_f[np.isnan(self.x_f)] = NAN_TOKEN

    @property
    def t(self) -> int:
        """The number of targets"""
        return len(self.y) if self.y is not None else 0


class _MolGraphDatasetMixin:
    data: list[_DatapointMixin]

    def __post_init__(self):
        self.reset()

    def __len__(self) -> int:
        return len(self.data)

    @cached_property
    def _Y(self) -> np.ndarray:
        """The raw targets of the dataset"""
        return np.array([d.y for d in self.data], float)

    @property
    def Y(self) -> np.ndarray:
        """The (scaled) targets of the dataset"""
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        self._validate_attribute(Y, "targets")

        self.__Y = np.array(Y, float)

    @cached_property
    def _X_f(self) -> np.ndarray:
        """The raw molecule features of the dataset"""
        return np.array([d.x_f for d in self.data])

    @property
    def X_f(self) -> np.ndarray:
        """The (scaled) molecule features of the dataset"""
        return self.__X_f

    @X_f.setter
    def X_f(self, X_f: ArrayLike):
        self._validate_attribute(X_f, "molecule features")

        self.__X_f = np.array(X_f)

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight for d in self.data])

    @property
    def gt_mask(self) -> np.ndarray:
        return np.array([d.gt_mask for d in self.data])

    @property
    def lt_mask(self) -> np.ndarray:
        return np.array([d.lt_mask for d in self.data])

    @property
    def t(self) -> int:
        return self.data[0].t

    @property
    def d_xf(self) -> int:
        """The extra molecule feature dimension, if any"""
        return 0 if np.equal(self.X_f, None).all() else self.X_f.shape[1]

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Normalizes the targets of this dataset using a :obj:`StandardScaler`

        The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
        each task independently.

        .. note::
            This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """
        if scaler is None:
            scaler = StandardScaler()
            self.Y = scaler.fit_transform(self._Y)
        else:
            self.Y = scaler.transform(self._Y)

        return scaler

    def normalize_inputs(
        self, key: str | None = "X_f", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_f", None}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        X = self.X_f

        if scaler is None:
            scaler = StandardScaler().fit(X)

        return scaler

    def reset(self):
        """Reset the {atom, bond, molecule} features and targets of each datapoint to its
        initial, unnormalized values.
        """
        self.__Y = self._Y
        self.__X_f = self._X_f

    def _validate_attribute(self, X: np.ndarray, label: str):
        if not len(self.data) == len(X):
            raise ValueError(
                f"number of molecules ({len(self.data)}) and {label} ({len(X)}) "
                "must have same length!"
            )
