import logging

import hydra
# import lightning as L
from omegaconf import DictConfig
from rich import print
# import torch.nn as nn

# import mol_gnn.cli.utils.instantiate as instantiate
from mol_gnn.data.dataset import Dataset
from mol_gnn.lightning_models.model import SimpleModel

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # modules = {
    #     name: instantiate.modules(module_config)
    #     for name, module_config in cfg.model.modules.items()
    # }
    # print(modules)
    # loss_configs = {
    #     name: instantiate.loss_config(loss_config) for name, loss_config in cfg.model.losses.items()
    # }
    # metric_configs = {
    #     name: instantiate.loss_config(metric_config)
    #     for name, metric_config in cfg.model.metrics.items()
    # }
    # print(lr_sched_factory)
    # print(optim)
    # print(lr_sched_factory(optim))
    # if cfg.model.lr_sched_config is not None:
    #     lr_sched_factory = meta_lr_sched_factory(lr_sched_factory, dict(cfg.model.lr_sched_config))

    # print(hydra.utils.instantiate(cfg.model.modules))
    # print(hydra.utils.instantiate(cfg.model.losses))
    # print(hydra.utils.instantiate(cfg.model.metrics))
    dataset: Dataset = hydra.utils.instantiate(cfg.data, _convert_="object")

    model: SimpleModel = hydra.utils.instantiate(cfg.model, _convert_="object")
    print(model)
    # model = hydra.utils.get_class(cfg.model._target_)(
    #     module_configs, loss_configs, metric_configs, optim_factory, lr_sched_factory
    # )
    # print(model)
    # trainer = L.Trainer()

    # hydra.utils.get_object(cfg.model)

    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")


if __name__ == "__main__":
    train()
