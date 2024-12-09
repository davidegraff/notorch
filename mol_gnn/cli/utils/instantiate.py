from collections.abc import Iterable, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from mol_gnn.types import LossConfig, ModuleConfig, TensorDictKey


def instantiate_in_keys(in_keys: Iterable[TensorDictKey] | Mapping[TensorDictKey, str]):
    if isinstance(in_keys, Mapping):
        return {tuple(key.split(".")): kwarg for key, kwarg in in_keys.items()}

    return [tuple(key.split(".")) for key in in_keys]


def instantiate_module_config(module_config: DictConfig) -> ModuleConfig:
    return ModuleConfig(
        hydra.utils.instantiate(module_config["module"]),
        instantiate_in_keys(OmegaConf.to_container(module_config["in_keys"])),
        module_config["out_keys"],
    )


def instantiate_loss_config(loss_config) -> LossConfig:
    return LossConfig(
        loss_config["weight"] if loss_config["weight"] is not None else 1.0,
        hydra.utils.instantiate(loss_config["module"]),
        instantiate_in_keys(OmegaConf.to_container(loss_config["in_keys"])),
    )


module_config = instantiate_module_config
loss_config = instantiate_loss_config
in_keys = instantiate_in_keys
