from omegaconf import OmegaConf
import pandas as pd


def register_resolvers():
    OmegaConf.register_new_resolver("csv", lambda path: pd.read_csv(path))
    OmegaConf.register_new_resolver("parquet", lambda path: pd.read_parquet(path))
