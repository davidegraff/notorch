import hydra
from omegaconf import DictConfig, OmegaConf

import mol_gnn
from mol_gnn.types import LRSchedConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    print(cfg.keys())
    print(cfg.model.lr_sched_factory)
    print(OmegaConf.to_yaml(cfg.model.lr_sched_factory))


if __name__ == "__main__":
    train()
