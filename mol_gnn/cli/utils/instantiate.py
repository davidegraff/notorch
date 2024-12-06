from collections.abc import Iterable, Mapping

import hydra
from omegaconf import OmegaConf

from mol_gnn.types import ModelModuleConfig, TensorDictKey


def instantiate_in_keys(in_keys: Iterable[TensorDictKey] | Mapping[TensorDictKey, str]):
    if isinstance(in_keys, Mapping):
        return {tuple(key.split(".")): kwarg for key, kwarg in in_keys.items()}

    return [tuple(key.split(".")) for key in in_keys]


def instantiate_module_config(module_config) -> ModelModuleConfig:
    return {
        "module": hydra.utils.instantiate(module_config["module"]),
        "in_keys": instantiate_in_keys(OmegaConf.to_container(module_config["in_keys"])),
        "out_keys": module_config[
            "out_keys"
        ],  # [tuple(key.split(".")) for key in module_config["out_keys"]],
    }


module_config = instantiate_module_config
in_keys = instantiate_in_keys
