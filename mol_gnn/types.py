# ruff: noqa: F401
from typing import Callable, TypedDict
from rdkit.Chem import Mol, Atom, Bond
from torch import Tensor, nn

type Rxn = tuple[Mol, Mol]
type TensorDictKey = tuple[str, ...] | str


class ModelModuleConfig(TypedDict):
    module: Callable
    in_keys: list[TensorDictKey] | dict[TensorDictKey, str]
    out_keys: list[TensorDictKey]


class LossModuleConfig(TypedDict):
    weight: float
    module: Callable[..., Tensor]
    in_keys: list[TensorDictKey] | dict[TensorDictKey, str]
