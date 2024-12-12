from collections.abc import Collection
from dataclasses import dataclass
import textwrap

from mol_gnn.conf import REPR_INDENT
from mol_gnn.databases.base import Database
from mol_gnn.transforms.base import Transform


@dataclass
class TransformManager[S, T, T_batched]:
    """A :class:`ManagedTransform` wraps around an input :class:`Transform` that reads and writes
    from to an input dictionary.

    It's like a :class:`~tensordict.nn.TensorDictModule` analog for :class:`Transform`s.
    """

    transform: Transform[S, T, T_batched]
    in_key: str
    out_key: str

    def collate(self, samples: Collection[dict]) -> T_batched:
        inputs = [sample[self.out_key] for sample in samples]

        return self.transform.collate(inputs)

    def __call__(self, sample: dict) -> dict:
        sample[self.out_key] = self.transform(sample[self.in_key])

        return sample

    def __repr__(self) -> str:
        text = "\n".join([
            f"(transform): {self.transform}",
            f"(in_key): {repr(self.in_key)}",
            f"(out_key): {repr(self.out_key)}"
        ])

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])


@dataclass
class DatabaseManager[KT, VT, T_batched]:
    """A :class:`ManagedTransform` wraps around an input :class:`Transform` that reads and writes
    from to an input dictionary.

    It's like a :class:`~tensordict.nn.TensorDictModule` analog for :class:`Transform`s.
    """

    db: Database[KT, VT]
    in_key: str
    out_key: str

    def __getitem__(self, sample: dict) -> dict:
        sample[self.out_key] = self.db[sample[self.in_key]]

        return sample

    def collate(self, samples: Collection[dict]) -> T_batched:
        inputs = [sample[self.out_key] for sample in samples]

        return self.db.collate(inputs)

    def __repr__(self) -> str:
        text = "\n".join([
            f"(database): {self.db}",
            f"(in_key): {repr(self.in_key)}",
            f"(out_key): {repr(self.out_key)}"
        ])

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])
