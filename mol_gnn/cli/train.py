import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from rich import print
from torch import nn

from mol_gnn.schedulers import meta_lr_sched_factory
from mol_gnn.types import LRSchedConfig
from mol_gnn.nn.gnn.embed import GraphEmbedding

log = logging.getLogger(__name__)

hydra.conf.ConfigStore
@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    optim_factory = hydra.utils.instantiate(cfg.model.optim_factory)
    optim = optim_factory(nn.Linear(10, 10).parameters())
    log.info(optim)

    lr_sched_factory = hydra.utils.instantiate(cfg.model.lr_sched_factory)
    # log.info(lr_sched)

    if cfg.model.lr_sched_config is not None:
        lr_sched_factory = meta_lr_sched_factory(lr_sched_factory, dict(cfg.model.lr_sched_config))

    log.info(f"{lr_sched_factory}")
    log.info(f"{lr_sched_factory(optim)}")

    model_config = {
        name: {
            "module": hydra.utils.instantiate(module_conf["module"]),
        }
        for name, module_conf in cfg.model.modules.items()
    }
    print(model_config)
    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")

if __name__ == "__main__":
    train()
