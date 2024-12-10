from collections.abc import Iterable, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf

from mol_gnn.types import LossConfig, ModuleConfig, TensorDictKey


def instantiate_in_keys(in_keys: Iterable[TensorDictKey] | Mapping[TensorDictKey, str]):
    if isinstance(in_keys, Mapping):
        return {tuple(key.split(".")): kwarg for key, kwarg in in_keys.items()}

    return [tuple(key.split(".")) for key in in_keys]


def _instantiate_module(module_config: DictConfig) -> ModuleConfig:
    return dict(
        module=hydra.utils.instantiate(module_config["module"]),
        in_keys=instantiate_in_keys((module_config["in_keys"])),
        out_keys=module_config["out_keys"],
    )


def _instantiate_losses(loss_config: DictConfig) -> LossConfig:
    return dict(
        module=hydra.utils.instantiate(loss_config["module"]),
        in_keys=instantiate_in_keys(OmegaConf.to_container(loss_config["in_keys"])),
        weight=loss_config.get("weight", 1.0),
    )


modules = _instantiate_module
losses = _instantiate_losses
