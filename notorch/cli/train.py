import logging
from typing import Callable

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from rich import print

from notorch.cli.utils.utils import build_group_transform_configs
from notorch.cli.utils.resolvers import register_resolvers
from notorch.data.dataset import NotorchDataset
from notorch.data.datamodule import NotorchDataModule
from notorch.lightning_models.model import NotorchModel

register_resolvers()
log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # print(dict(**cfg.dataloader))
    # print(OmegaConf.to_yaml(cfg))
    # print(hydra.utils.instantiate(cfg.data, _convert_="object"))
    # import pdb; pdb.set_trace()
    # data_cfg = hydra.utils.instantiate(cfg.data, _convert_="object")
    # print(data_cfg)
    # data_cfg["df"] = df
    # dataset_factory: Callable[..., NotorchDataset] = hydra.utils.instantiate(
    #     cfg.data, _convert_="object"
    # )
    train: NotorchDataset = hydra.utils.instantiate(cfg.train, _convert_="object")
    val: NotorchDataset = hydra.utils.instantiate(cfg.val, _convert_="object")
    transform_key_map = hydra.utils.instantiate(cfg.model.transforms)
    target_transforms = train.build_task_transform_configs()
    transforms = build_group_transform_configs(transform_key_map, target_transforms)
    model: NotorchModel = hydra.utils.instantiate(
        cfg.model, transforms=transforms, _convert_="object"
    )

    print(model)

    trainer = L.Trainer(accelerator="cpu")
    train_loader = train.to_dataloader(**cfg.dataloader, shuffle=True)
    val_loader = val.to_dataloader(**cfg.dataloader)
    trainer.fit(model, train_loader, val_loader)

    # print(dataset)
    # transforms = (hydra.utils.instantiate(cfg.data.transforms, _convert_="object"))
    # target_groups = (hydra.utils.instantiate(cfg.data.target_groups, _convert_="object"))
    # print(NotorchDataset(None, transforms, {"foo": "bar"}, target_groups))
    # dataset: Dataset = hydra.utils.instantiate(cfg.data, _convert_="object")

    # print(model)
    # trainer = L.Trainer()

    # hydra.utils.get_object(cfg.model)

    # for name, module_conf in cfg.model.modules.items():
    #     print(f"{name} ---- {module_conf}")


if __name__ == "__main__":
    train()
