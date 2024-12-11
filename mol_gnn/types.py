# ruff: noqa: F401
from typing import Callable, Literal, NamedTuple, Required, TypedDict

from rdkit.Chem import Atom, Bond, Mol
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

type Rxn = tuple[Mol, Mol]


class TransformConfig(TypedDict):
    transform: Callable
    in_key: str
    out_key: str


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
