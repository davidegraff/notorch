import logging
from typing import Callable

import hydra
import lightning as L
from omegaconf import DictConfig
from rich import print

from notorch.data.dataset import NotorchDataset
from notorch.lightning_models.model import NotorchModel
from notorch.types import TargetTransformConfig, TaskTransformConfig
from notorch.cli.utils.resolvers import register_resolvers

register_resolvers()
log = logging.getLogger(__name__)


def build_transform_config(
    transform_key_map: dict[str, dict[str, str]], task_transforms: dict[str, TaskTransformConfig]
) -> dict[str, TargetTransformConfig]:
    return {
        target_group: {
            mode: {
                "module": task_transforms[target_group][mode],
                "key": transform_key_map[target_group][mode],
            }
            for mode in ["preds", "targets"]
            if mode in transform_key_map[target_group]
        }
        for target_group in (transform_key_map or dict()).keys()
    }


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    # print(hydra.utils.instantiate(cfg.data, _convert_="object"))
    # import pdb; pdb.set_trace()
    # data_cfg = hydra.utils.instantiate(cfg.data, _convert_="object")
    # print(data_cfg)
    # data_cfg["df"] = df
    dataset_factory: Callable[..., NotorchDataset] = hydra.utils.instantiate(
        cfg.data, _convert_="object"
    )
    train = dataset_factory(cfg.train_df)
    val = dataset_factory(cfg.train_df)

    transform_key_map = hydra.utils.instantiate(cfg.model.transforms)
    target_transforms = train.build_task_transform_configs()
    transforms = build_transform_config(transform_key_map, target_transforms)
    model: NotorchModel = hydra.utils.instantiate(
        cfg.model, transforms=transforms, _convert_="object"
    )

    # print(model_kwargs)
    print(model)

    # print(train)
    # print(len(train))
    # print(train[4])

    trainer = L.Trainer(accelerator="cpu")
    train_loader = train.to_dataloader(batch_size=64)
    trainer.fit(model, train_loader)

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
