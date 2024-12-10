from collections.abc import Collection, Iterable, Mapping
from copy import copy
from dataclasses import InitVar, dataclass, field
from typing import Any

# from jaxtyping import Array, Float
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tensordict import TensorDict

from mol_gnn.transforms.base import ManagedTransform, JoinColumns
from mol_gnn.types import TransformConfig


@dataclass
class Dataset(Dataset[dict]):
    df: InitVar[pd.DataFrame]
    transforms: Mapping[str, ManagedTransform]
    target_groups: Mapping[str, list[str]]
    # extra_data: Mapping[str, Float[Array, "n d"]]

    records: list[dict[str, Any]] = field(init=False)

    def __post_init__(self, df: pd.DataFrame):
        for group_name, columns in self.target_groups.items():
            df[group_name] = df[columns].values.tolist()

        columns = [transform.in_key for transform in self.transforms.values()] + list(
            self.target_groups.keys()
        )
        self.records = df[columns].to_dict("records")

    def __getitem__(self, idx: int) -> dict:
        record = copy(self.records[idx])
        for transform in self.transforms.values():
            record = transform(record)

        return record
        # dicts = [transform(record) for transform in self.transforms.values()]
        # out = reduce(lambda a, b: a | b, dicts, record)

        extra_transform_data = {k: v for k, v in self.extra_transforms.items()}
        extra_data = {key: value[idx] for key, value in self.extra_data.items()}

        return sample_data | extra_data | extra_transform_data

    def collate(self, samples: Collection[dict]) -> TensorDict:
        batch = TensorDict({}, batch_size=len(samples))

        for transform in self.transforms.values():
            batch["input", transform.out_key] = transform.collate(samples)
        for group_name in self.target_groups:
            batch["target", group_name] = torch.as_tensor(
                [sample[group_name] for sample in samples]
            )

        return batch

    def to_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate, **kwargs)
