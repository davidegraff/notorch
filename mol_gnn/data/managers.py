from collections.abc import Collection
from dataclasses import dataclass
import textwrap
from typing import Protocol

from mol_gnn.conf import REPR_INDENT
from mol_gnn.databases.base import Database
from mol_gnn.transforms.base import Transform


class Manager(Protocol):
    """A :class:`Manager` manages an input :attr:`asset`."""

    in_key: str
    out_key: str

    def update(self, sample: dict) -> dict: ...
    def collate(self, samples: Collection[dict]): ...


@dataclass
class TransformManager(Manager):
    """A :class:`TransformManager` wraps around an input :class:`Transform` that reads and writes
    from to an input dictionary.

    It's like a :class:`~tensordict.nn.TensorDictModule` analog for :class:`Transform`s.
    """

    transform: Transform
    in_key: str = None
    out_key: str = None

    def __post_init__(self):
        if self.in_key is None:
            self.in_key = self.transform._in_key_
        if self.out_key is None:
            self.out_key = self.transform._out_key_

    def update(self, sample: dict) -> dict:
        sample[self.out_key] = self.transform(sample[self.in_key])

        return sample

    def collate(self, samples: Collection[dict]):
        try:
            inputs = [sample[self.out_key] for sample in samples]
        except KeyError:
            raise KeyError(
                f"arg 'samples' is missing key input key '{self.in_key}'! "
                f"Is this input the result of `{type(self).__name__}.update()`?"
            )

        return self.transform.collate(inputs)


    def __repr__(self) -> str:
        text = "\n".join(
            [
                f"(transform): {self.transform}",
                f"(in_key): {repr(self.in_key)}",
                f"(out_key): {repr(self.out_key)}",
            ]
        )

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])


@dataclass
class DatabaseManager(Manager):
    database: Database
    in_key: str
    out_key: str

    def update(self, sample: dict) -> dict:
        sample[self.out_key] = self.database[sample[self.in_key]]

        return sample

    def collate(self, samples: Collection[dict]):
        try:
            inputs = [sample[self.out_key] for sample in samples]
        except KeyError:
            raise KeyError(
                f"arg 'samples' is missing key input key '{self.in_key}'! "
                f"Is this input the result of `{type(self).__name__}.update()`?"
            )

        return self.database.collate(inputs)


    def __repr__(self) -> str:
        text = "\n".join(
            [
                f"(database): {self.database}",
                f"(in_key): {repr(self.in_key)}",
                f"(out_key): {repr(self.out_key)}",
            ]
        )

        return "\n".join([f"{type(self).__name__}(", textwrap.indent(text, REPR_INDENT), ")"])
