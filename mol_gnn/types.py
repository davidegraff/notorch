# ruff: noqa: F401
from rdkit.Chem import Mol, Atom, Bond
from torch import nn

type Rxn = tuple[Mol, Mol]

type TensorDictKey = tuple[str, ...] | str
type ModelConfig = dict[
    str, tuple[nn.Module, list[TensorDictKey] | dict[TensorDictKey, str], list[TensorDictKey]]
]
type LossConfig = dict[str, tuple[float, nn.Module, list[TensorDictKey] | dict[TensorDictKey, str]]]
