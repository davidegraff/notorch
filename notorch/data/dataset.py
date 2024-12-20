from collections.abc import Collection, Mapping
from copy import copy
import textwrap

import pandas as pd
from rich.pretty import pretty_repr
from tensordict import TensorDict
import torch
from torch.utils.data import DataLoader, Dataset

from notorch.conf import INPUT_KEY_PREFIX, REPR_INDENT, TARGET_KEY_PREFIX
from notorch.data.managers import DatabaseManager, TransformManager
from notorch.nn.transforms import build as build_task_transforms
from notorch.types import DatabaseConfig, TargetConfig, TaskTransformConfig


class NotorchDataset(Dataset[dict]):
    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Mapping[str, TaskTransformConfig],
        target_groups: Mapping[str, TargetConfig],
        databases: Mapping[str, DatabaseConfig] | None = None,
    ):
        databases = {
            name: DatabaseManager(**kwargs) for name, kwargs in (databases or dict()).items()
        }

        self.df = df
        self.transforms = {name: TransformManager(**kwargs) for name, kwargs in transforms.items()}
        self.target_groups = target_groups
        self.databases = databases

        # transform_columns = list(set(transform.in_key for transform in self.transforms.values()))
        # db_columns = list(set(db.in_key for db in self.databases.values()))
        # columns = transform_columns + db_columns
        self.records = self.df.to_dict("records")
        self.targets = {
            name: torch.as_tensor(self.df[config["columns"]].values).to(torch.float)
            for name, config in self.target_groups.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        sample = copy(self.records[idx])

        for name, db in self.databases.items():
            sample = db.update(sample)
        for transform in self.transforms.values():
            sample = transform.update(sample)
        for name, targets in self.targets.items():
            sample[name] = targets[idx]

        return sample

    def collate(self, samples: Collection[dict]) -> TensorDict:
        batch = TensorDict({}, batch_size=len(samples))

        for transform in self.transforms.values():
            batch[f"{INPUT_KEY_PREFIX}.{transform.out_key}"] = transform.collate(samples)
        for db in self.databases.values():
            batch[f"{INPUT_KEY_PREFIX}.{db.out_key}"] = db.collate(samples)
        for name in self.target_groups:
            batch[f"{TARGET_KEY_PREFIX}.{name}"] = torch.stack(
                [sample[name] for sample in samples], dim=0
            )

        return batch

    def to_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate, **kwargs)

    def build_task_transform_configs(self) -> dict[str, TaskTransformConfig]:
        """Build a mapping from target group name to its respective :class:`TaskTransformConfig`."""
        return {
            name: build_task_transforms(config.get("task"), self.targets[name])
            for name, config in self.target_groups.items()
        }

    # def __enter__(self) -> Self:
    #     self.stack = ExitStack()
    #     [self.stack.enter_context(db) for db in self.databases.values()]

    #     return self

    # def __exit__(self, *exc):
    #     self.stack = self.stack.close()

    def __repr__(self) -> str:
        prettify = lambda obj: pretty_repr(obj, indent_size=2, max_length=4)  # noqa: E731
        df_repr = f"(records): {prettify(self.records)}"
        transform_repr = "\n".join(
            [
                "(transforms): {",
                textwrap.indent(
                    "\n".join(
                        f"({name}): {prettify(transform)}"
                        for name, transform in self.transforms.items()
                    ),
                    REPR_INDENT,
                ),
                "}",
            ]
        )
        databases_repr = "\n".join(
            [
                "(databases): {",
                textwrap.indent(
                    "\n".join(f"({name}): {db}" for name, db in self.databases.items()), REPR_INDENT
                ),
                "}",
            ]
        )
        databases_repr = "\n".join([f"(databases): {prettify(self.databases)}"])
        target_groups_repr = "\n".join([f"(target_groups): {prettify(self.target_groups)}"])

        return "\n".join(
            [
                f"{type(self).__name__}(",
                textwrap.indent(df_repr, REPR_INDENT),
                textwrap.indent(transform_repr, REPR_INDENT),
                textwrap.indent(databases_repr, REPR_INDENT),
                textwrap.indent(target_groups_repr, REPR_INDENT),
                ")",
            ]
        )


"""
NotorchDataset(
  (transforms):
    'smi_to_mol': SmiToMol(keep_h=True, add_hs=False)
    'smi_to_graph': Pipeline(
      (0): SmiToMol(...)
      (1): MolToGraph(
        (atom_transform): MultiTypeAtomTransform(
          (elements): [...]
          (num_hs): [...]
        )
        (bond_transform): MultiTypeBondTransform(
          (bond_types): [...]
          (stereos): [...]
        )
      )
    )
  )
  (databases):
    (qm_descs):
  (target_groups): {
    'regression': ['a', 'b', 'c']
    'classification': ['d', 'e']
  }
)
"""
