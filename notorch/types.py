# ruff: noqa: F401
from collections.abc import Mapping
from typing import Callable, Literal, NamedTuple, Protocol, Required, TypedDict

from rdkit.Chem import Mol
from torch import Tensor

from notorch.databases.base import Database

type Rxn = tuple[Mol, Mol]
TaskType = Literal["regression", "classification", "multiclass", "mve", "evidential", "dirichlet"]


class DatabaseConfig(TypedDict):
    db: Database
    in_key: str
    out_key: str


class TaskTransformConfig(TypedDict):
    preds: Callable | None
    targets: Callable | None


class TransformConfig(TypedDict):
    module: Callable | None
    key: str


class GroupTransformConfig(TypedDict):
    preds: TransformConfig
    targets: TransformConfig


class LossWeightConfig(TypedDict):
    train: Mapping[str, float] | None
    val: Mapping[str, float] | None


class TargetConfig(TypedDict, total=False):
    task: Required[TaskType]
    weight: float


class ModuleConfig(TypedDict):
    module: Callable
    in_keys: list[str] | dict[str, str]
    out_keys: list[str]


class LossConfig(TypedDict):
    module: Callable[..., Tensor]
    in_keys: list[str] | dict[str, str]
    weight: float


Reduction = Literal["mean", "sum", "min", "max"]
