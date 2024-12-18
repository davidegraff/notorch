import logging

import hydra

import numpy as np
from omegaconf import DictConfig
from rich import print
import pandas as pd

from mol_gnn.data.dataset import NotorchDataset
from mol_gnn.types import TargetTransformConfig

log = logging.getLogger(__name__)


def build_transform_config(
    transform_key_map, target_transforms
) -> dict[str, TargetTransformConfig]:
    return {
        target_group: {
            mode: {
                "module": target_transforms[target_group][mode],
                "key": transform_key_map[target_group][mode],
            }
            for mode in ["preds", "targets"]
            if mode in transform_key_map[target_group]
        }
        for target_group in transform_key_map.keys()
    }


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    X = np.random.randn(5, 128)
    df = pd.DataFrame(dict(zip("abcde", X)))
    # print(hydra.utils.instantiate(cfg.data, _convert_="object"))
    # import pdb; pdb.set_trace()
    data_cfg = hydra.utils.instantiate(cfg.data, _convert_="object")
    data_cfg["df"] = df
    dataset = NotorchDataset(**data_cfg)

    target_transforms = dataset.build_task_transform_configs()
    transform_key_map = hydra.utils.instantiate(cfg.model.transforms)
    transforms = build_transform_config(transform_key_map, target_transforms)
    model: SimpleModel = hydra.utils.instantiate(
        cfg.model, transforms=transforms, _convert_="object"
    )
    print(model)

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
