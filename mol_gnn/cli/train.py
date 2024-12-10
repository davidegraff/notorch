import logging

import hydra
import lightning as L
from omegaconf import DictConfig
from rich import print

import mol_gnn.cli.utils.instantiate as instantiate
from mol_gnn.data.dataset import Dataset
from mol_gnn.schedulers import meta_lr_sched_factory

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    dataset: Dataset = hydra.utils.instantiate(cfg.data, _convert_="object")

    module_configs = {
        name: instantiate.module_config(module_config)
        for name, module_config in cfg.model.modules.items()
    }
    loss_configs = {
        name: instantiate.loss_config(loss_config) for name, loss_config in cfg.model.losses.items()
    }
    metric_configs = {
        name: instantiate.loss_config(metric_config)
        for name, metric_config in cfg.model.metrics.items()
    }
    optim_factory = hydra.utils.instantiate(cfg.model.optim_factory)
    lr_sched_factory = hydra.utils.instantiate(cfg.model.lr_sched_factory)
    if cfg.model.lr_sched_config is not None:
        lr_sched_factory = meta_lr_sched_factory(lr_sched_factory, dict(cfg.model.lr_sched_config))

    model = hydra.utils.get_class(cfg.model._target_)(
        module_configs, loss_configs, metric_configs, optim_factory, lr_sched_factory
    )
    print(model)
    trainer = L.Trainer()

    # hydra.utils.get_object(cfg.model)

    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")


if __name__ == "__main__":
    train()
