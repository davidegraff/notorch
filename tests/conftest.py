from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def smis():
    return pd.read_csv(DATA_DIR / "smis.csv")['smiles'].tolist()


@pytest.fixture
def rxns():
    return pd.read_csv(DATA_DIR / "rxns.csv")['rxn'].tolist()


@pytest.fixture
def multi():
    return pd.read_csv(DATA_DIR / "smis.csv").values.tolist()