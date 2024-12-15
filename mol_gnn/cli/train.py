import logging

import hydra

# import lightning as L
from omegaconf import DictConfig
from rich import print
# import torch.nn as nn

from mol_gnn.data.dataset import NotorchDataset

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # print(hydra.utils.instantiate(cfg.model.modules))
    # print(hydra.utils.instantiate(cfg.model.losses))
    # print(hydra.utils.instantiate(cfg.model.metrics))
    transforms = (hydra.utils.instantiate(cfg.data.transforms, _convert_="object"))
    target_groups = (hydra.utils.instantiate(cfg.data.target_groups, _convert_="object"))
    print(NotorchDataset(None, transforms, {"foo": "bar"}, target_groups))
    # dataset: Dataset = hydra.utils.instantiate(cfg.data, _convert_="object")

    # model: SimpleModel = hydra.utils.instantiate(cfg.model, _convert_="object")
    # print(model)
    # trainer = L.Trainer()

    # hydra.utils.get_object(cfg.model)

    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")


if __name__ == "__main__":
    train()
