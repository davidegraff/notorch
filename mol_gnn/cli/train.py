import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from rich import print
from tensordict.nn import TensorDictModule, TensorDictSequential

import mol_gnn.cli.utils.instantiate as instantiate
from mol_gnn.schedulers import meta_lr_sched_factory

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # optim_factory = hydra.utils.instantiate(cfg.model.optim_factory)
    lr_sched_factory = hydra.utils.instantiate(cfg.model.lr_sched_factory)
    if cfg.model.lr_sched_config is not None:
        lr_sched_factory = meta_lr_sched_factory(lr_sched_factory, dict(cfg.model.lr_sched_config))

    modules_config = {
        name: instantiate.module_config(module_config)
        for name, module_config in cfg.model.modules.items()
    }

    # modules = [
    #     TensorDictModule(
    #         module_config["module"],
    #         module_config["in_keys"],
    #         [(name, key) for key in module_config["out_keys"]],
    #     )
    #     for name, module_config in modules_config.items()
    # ]
    modules = [
        TensorDictModule(
            module_config["module"],
            module_config["in_keys"],
            module_config["out_keys"],
        )
        for name, module_config in modules_config.items()
    ]

    model = TensorDictSequential(*modules)

    print(modules_config)
    # print(modules)
    print(model)

    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")


if __name__ == "__main__":
    train()
