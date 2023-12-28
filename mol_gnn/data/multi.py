from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from mol_gnn.data.batch import MultiInputMpnnBatch
from mol_gnn.data.mixins import _MolGraphDatasetMixin, Datum
from mol_gnn.data.molecule import MoleculeDataset
from mol_gnn.data.reaction import ReactionDataset



@dataclass(repr=False, eq=False)
class MultiInputDataset(Dataset):
    """A :class:`MulticomponentDataset` is a :class:`Dataset` composed of parallel
    :class:`MoleculeDataset`s and :class:`ReactionDataset`s"""

    datasets: list[MoleculeDataset | ReactionDataset]

    def __post_init__(self):
        sizes = [len(dset) for dset in self.datasets]
        if not (np.diff(sizes) == 0).all():
            raise ValueError(f"datasets must have all same length! got: {sizes}")

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> list[Datum]:
        return [dset[idx] for dset in self.datasets]

    @property
    def smiles(self) -> list[list[str]]:
        return list(zip(*[dset.smiles for dset in self.datasets]))

    @property
    def mols(self) -> list[list[Chem.Mol]]:
        return list(zip(*[dset.mols for dset in self.datasets]))

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        return self.datasets[0].normalize_targets(scaler)

    def normalize_inputs(
        self, key: str | None = "X_f", scaler: StandardScaler | None = None
    ) -> list[StandardScaler]:
        return [dset.normalize_inputs(key, scaler) for dset in self.datasets]

    def reset(self):
        return [dset.reset() for dset in self.datasets]

    def collate_batch(batches: Iterable[Iterable[Datum]]) -> MultiInputMpnnBatch:
        input_batches = [_MolGraphDatasetMixin.collate_batch(batch) for batch in zip(*batches)]

        return MultiInputMpnnBatch(
            [batch.G for batch in input_batches],
            [batch.V_d for batch in input_batches],
            input_batches[0].X_f,
            input_batches[0].Y,
            input_batches[0].w,
            input_batches[0].lt_mask,
            input_batches[0].gt_mask,
        )

    def to_dataloader(self, batch_size: int = 128, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size, collate_fn=self.collate_batch, **kwargs)
