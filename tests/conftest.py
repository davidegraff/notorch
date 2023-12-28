from pathlib import Path
import numpy as np

from rdkit import Chem
import pandas as pd
import pytest

from mol_gnn.data.molecule import MoleculeDatapoint

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def smis():
    return pd.read_csv(DATA_DIR / "smis.csv")['smiles'].tolist()


@pytest.fixture
def mols(smis: list):
    return [Chem.MolFromSmiles(smi) for smi in smis]


@pytest.fixture
def random_mol_data(mols: list):
    Y = np.random.randn(len(mols), 1)
    data = [MoleculeDatapoint(mol, y) for mol, y in zip(mols, Y)]

    return data


@pytest.fixture
def lipo_data():
    df = pd.read_csv(DATA_DIR / "lipo.csv")
    data = [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(df['smiles'], df[['lipo']].values)]

    return data


@pytest.fixture
def rxns():
    return pd.read_csv(DATA_DIR / "rxns.csv")['rxn'].tolist()


@pytest.fixture
def multi():
    return pd.read_csv(DATA_DIR / "smis.csv").values.tolist()