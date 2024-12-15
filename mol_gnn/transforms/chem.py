from dataclasses import dataclass
from typing import ClassVar

from rdkit import Chem

from mol_gnn.transforms.base import Transform
from mol_gnn.types import Mol

SANITIZE_OPS = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS


@dataclass
class SmiToMol(Transform[str, Mol, list[Mol]]):
    _in_key_: ClassVar[str] = "smi"
    _out_key_: ClassVar[str] = "mol"

    keep_h: bool = True
    add_h: bool = False

    def __call__(self, smi: str) -> Mol:
        if self.keep_h:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=SANITIZE_OPS)
        else:
            mol = Chem.MolFromSmiles(smi)

        return Chem.AddHs(mol) if self.add_h else mol

    collate = list
