# ruff: noqa: F401
from collections.abc import Collection
from typing import Callable, Literal, NamedTuple, Protocol, Required, TypedDict

from rdkit.Chem import Atom, Bond, Mol
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from mol_gnn.databases.base import Database
from mol_gnn.transforms.base import Transform

type Rxn = tuple[Mol, Mol]
TaskType = Literal[
    "regression", "classification", "multiclass", "Evidential", "mve", "evidential", "dirichlet"
]


class DatabaseConfig(TypedDict):
    db: Database
    in_key: str
    out_key: str


class TransformConfig(TypedDict):
    transform: Transform
    in_key: str
    out_key: str


class TargetConfig(TypedDict):
    columns: list[str]
    task: str


class ModuleConfig(TypedDict):
    module: Callable
    in_keys: list[str] | dict[str, str]
    out_keys: list[str]


class LossConfig(TypedDict):
    module: Callable[..., Tensor]
    in_keys: list[str] | dict[str, str]
    weight: float


class LRSchedConfig(TypedDict, total=False):
    scheduler: Required[LRScheduler]
    interval: Literal["step", "epoch"]
    frequency: int
    monitor: str
    strict: bool
    name: str | None
