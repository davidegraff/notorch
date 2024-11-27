from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Self

from rdkit import Chem

from mol_gnn.data.models.datum import Datum
from mol_gnn.types import Mol, Rxn
from mol_gnn.featurizers import VectorFeaturizer, GraphFeaturizer, CGRFeaturizer
from mol_gnn.data.datasets.molecule import _MolGraphDatasetMixin
from mol_gnn.data.datasets.mixins import _DatapointMixin
from mol_gnn.utils.chem import make_mol


@dataclass
class _ReactionDatapointMixin:
    rct: Mol
    """the reactant associated with this datapoint"""
    pdt: Mol
    """the product associated with this datapoint"""

    @classmethod
    def from_smi(
        cls,
        rxn_or_smis: str | tuple[str, str],
        *args,
        keep_h: bool = False,
        add_h: bool = False,
        **kwargs,
    ) -> Self:
        match rxn_or_smis:
            case str():
                rct_smi, agt_smi, pdt_smi = rxn_or_smis.split(">")
                rct_smi = f"{rct_smi}.{agt_smi}" if agt_smi else rct_smi
            case tuple():
                rct_smi, pdt_smi = rxn_or_smis
            case _:
                raise TypeError(
                    "Must provide either a reaction SMARTS string or a tuple of "
                    "reactant and product SMILES strings!"
                )

        rct = make_mol(rct_smi, keep_h, add_h)
        pdt = make_mol(pdt_smi, keep_h, add_h)

        return cls(rct, pdt, *args, **kwargs)


@dataclass
class ReactionDatapoint(_DatapointMixin, _ReactionDatapointMixin):
    """
    A :class:`ReactionDatapoint` contains a single reaction and its associated features and
    targets.
    """

    def __post_init__(self, mfs: list[VectorFeaturizer[Mol]] | None):
        if self.rct is None:
            raise ValueError("Reactant cannot be `None`!")
        if self.pdt is None:
            raise ValueError("Product cannot be `None`!")

        return super().__post_init__(mfs)

    def calc_features(self, mfs: list[VectorFeaturizer[Mol]]) -> np.ndarray:
        x_fs = [
            mf(mol) if mol.GetNumHeavyAtoms() > 0 else np.zeros(len(mf))
            for mf in mfs
            for mol in [self.rct, self.pdt]
        ]

        return np.hstack(x_fs)


@dataclass
class ReactionDataset(_MolGraphDatasetMixin):
    """A :class:`ReactionDataset` composed of :class:`ReactionDatapoint`s"""

    data: list[ReactionDatapoint]
    """the input data"""
    featurizer: GraphFeaturizer[Rxn] = field(default_factory=CGRFeaturizer)
    """the featurizer with which to generate MolGraphs of the input"""

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.featurizer((d.rct, d.pdt), None, None)

        return Datum(mg, None, d.x_f, d.y, d.weight, d.lt_mask, d.gt_mask)

    @property
    def smiles(self) -> list[tuple[str, str]]:
        return [(Chem.MolToSmiles(d.rct), Chem.MolToSmiles(d.pdt)) for d in self.data]

    @property
    def mols(self) -> list[tuple[Chem.Mol, Chem.Mol]]:
        return [(d.rct, d.pdt) for d in self.data]
