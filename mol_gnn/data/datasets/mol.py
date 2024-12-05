from dataclasses import dataclass, field

from jaxtyping import Array, ArrayLike, Float, Num
from rdkit import Chem
from torch.utils.data import Dataset

from mol_gnn.data.models.graph import Graph
from mol_gnn.data.models.sample import Sample
from mol_gnn.transforms.base import Transform
from mol_gnn.transforms.graph import MolToGraph
from mol_gnn.types import Mol


@dataclass
class MoleculeDataset(Dataset[Sample]):
    mols: list[Mol]
    Y: Num[ArrayLike, "n *t"]
    graph_transform: Transform[Mol, Graph] = field(default_factory=MolToGraph)
    extra_transforms: dict[str, Transform[Mol, Float[Array, "n d"]]]
    extra_data: dict[str, Float[Array, "n d"]]

    def __getitem__(self, idx: int) -> Sample:
        G = self.graph_transform(self.mols[idx])
        y = self.Y[idx]

        sample_data = dict(G=G, y=y)
        extra_transform_data = {k: v for k, v in self.extra_transforms.items()}
        extra_data = {key: value[idx] for key, value in self.extra_data.items()}

        return sample_data | extra_data | extra_transform_data

    @property
    def smiles(self) -> list[str]:
        """The SMILES strings associated with the dataset"""
        return [Chem.MolToSmiles(mol) for mol in self.mols]
