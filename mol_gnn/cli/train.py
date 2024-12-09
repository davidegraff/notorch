import logging

import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from omegaconf import DictConfig
from rich import print

import mol_gnn.cli.utils.instantiate as instantiate
from mol_gnn.schedulers import meta_lr_sched_factory

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # optim_factory = hydra.utils.instantiate(cfg.model.optim_factory)
    lr_sched_factory = hydra.utils.instantiate(cfg.model.lr_sched_factory)
    if cfg.model.lr_sched_config is not None:
        lr_sched_factory = meta_lr_sched_factory(lr_sched_factory, dict(cfg.model.lr_sched_config))

    module_configs = {
        name: instantiate.module_config(module_config)
        for name, module_config in cfg.model.modules.items()
    }
    loss_configs = {
        name: instantiate.loss_config(loss_config)
        for name, loss_config in cfg.model.losses.items()
    }
    metric_configs = {
        name: instantiate.loss_config(metric_config)
        for name, metric_config in cfg.model.metrics.items()
    }

    # modules = [
    #     TensorDictModule(
    #         module_config["module"], module_config["in_keys"], module_config["out_keys"]
    #     )
    #     for name, module_config in module_configs.items()
    # ]
    # model = TensorDictSequential(*modules)

    model = hydra.utils.instantiate(
        cfg.model, module_configs, loss_configs, metric_configs, optim_factory, lr_sched_factory
    )
    print(module_configs)
    # print(modules)
    print(model)

    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")


if __name__ == "__main__":
    train()
