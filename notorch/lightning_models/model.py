from __future__ import annotations

from collections.abc import Callable

import lightning as L
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT

from notorch.conf import TARGET_KEY_PREFIX
from notorch.types import GroupTransformConfig, LossConfig, LRSchedConfig, ModuleConfig


def is_target_key(key: str):
    """Is the input key :attr:`key` of the form ``targets.*``?"""
    return key.split(".")[0] == TARGET_KEY_PREFIX


class NotorchModel(L.LightningModule):
    """A :class:`SimpleModel` is a generic class for composing (mostly) arbitrary models.

    The general recipe consists of three configuration dictionaries that define the model, the loss,
    and the evaluation/testing metrics, respectively. The model configuration dictionary defines a

    Parameters
    ----------
    modules : dict[str, ModelModuleConfig]
        A mapping from a name to a dictionary with the keys:

        * ``module``: the :class:`~torch.nn.Module` that will be wrapped inside a
        :class:`~tensordict.nn.TensorDictModule`.
        * ``in_keys``: the input keys to the module as either:

            - a list of input keys that will be fetched from the intermediate `TensorDict`
            and passed in as positional arguments to the module
            - a dictionary mapping from keyword argument name to the keys that will be
            fetched and supplied to the corresponding argument

        * ``out_keys``: the keys under which the module's output will be placed into the tensordict
        .. note::
            The output values will be placed in a sub-tensordict under the module's name (i.e.,
            the key corresponding to the 3-tuple)

    losses : dict[str, LossModuleConfig]
        A mapping from a name to a dictionary with the keys:

        - ``weight``: a float for the term's weight in the total loss
        - ``module``: a callable that returns a single tensor
        - ``in_keys``: the input keys of the module

        .. note::
            Each term will be placed into the tensordict under the key `("loss.<NAME>")`

        The overall training loss is computed as the weighted sum of all terms. For more
        details on the ``in_keys`` key, see :attr:`modules`.

    metrics : dict[str, LossModuleConfig]
        A mapping from a name to a dictionary with the keys:

        - ``weight``: a float for the term's weight in the total validation loss
        - ``module``: a callable that returns a single tensor
        - ``in_keys``: the input keys of the module

        .. note::
            Each term will be placed into the tensordict under the key `("metric.<NAME>")`

        The overall validation loss is computed as the weighted sum of all loss term values. For
        details on the ``in_keys`` key, see :attr:`modules`.

    transforms : dict[str, TargetTransformConfig] | None
        A mapping from a name to a dictionary of dictionaries defining the configuration for both
        prediction and target transforms. The outer dictionary contains two keys, ``"preds"`` and
        ``"targets"``, each mapping to a dictionary with the following keys:

        - ``module``: any ``Callable``, typically a :class:`~torch.nn.Module`, that has **no
        learnable parameters** that will be applied to the specified key in the tensordict
        - ``key``: the key in the tensordict whose value will be _modified and ovewritten in place_.

        The ``"preds"`` transforms will be applied to model predictions at inference time via
        :meth:`SimpleModel.predict_step` and the ``"targets"`` transforms will be applied to the
        input targets during training and validation.

        .. note::
            In the event that the specified keys are not present in the tensordict, then the
            transforms will have no effect. As such, you must take care to ensure the keys have been
            named correctly.
    """

    def __init__(
        self,
        modules: dict[str, ModuleConfig],
        losses: dict[str, LossConfig],
        metrics: dict[str, LossConfig],
        transforms: dict[str, GroupTransformConfig] | None = None,
        optim_factory: Callable[[ParamsT], Optimizer] = Adam,
        lr_sched_factory: Callable[[Optimizer], LRScheduler | LRSchedConfig] | None = None,
        keep_all_output: bool = False,
    ):
        super().__init__()

        model_modules = [
            TensorDictModule(
                module_config["module"],
                module_config["in_keys"],
                [f"{name}.{key}" for key in module_config["out_keys"]],
            )
            for name, module_config in modules.items()
        ]

        selected_out_keys = set()
        loss_modules = []
        for name, loss_config in losses.items():
            module = TensorDictModule(
                loss_config["module"], loss_config["in_keys"], [f"loss.{name}"], inplace=False
            )
            module._weight = loss_config["weight"]
            loss_modules.append(module)
            selected_out_keys.update([k for k in loss_config["in_keys"] if not is_target_key(k)])
        metric_modules = []
        for name, metric_config in metrics.items():
            module = TensorDictModule(
                metric_config["module"], metric_config["in_keys"], [f"metric.{name}"], inplace=False
            )
            module._weight = metric_config["weight"]
            metric_modules.append(module)
            selected_out_keys.update([k for k in metric_config["in_keys"] if not is_target_key(k)])

        selected_out_keys = None if keep_all_output else list(selected_out_keys)

        transforms_dict = {"preds": [], "targets": []}
        for group_transform_config in (transforms or dict()).values():
            for mode in ["preds", "targets"]:
                if mode not in group_transform_config:
                    continue
                mod = group_transform_config[mode]["module"]
                key = group_transform_config[mode]["key"]
                if mod is None:
                    continue
                module = TensorDictModule(mod, [key], [key])
                transforms_dict[mode].append(module)
        transforms_dict = {
            key: TensorDictSequential(*modules, partial_tolerant=True)
            for key, modules in transforms_dict.items()
        }

        self.model = TensorDictSequential(*model_modules, selected_out_keys=selected_out_keys)
        self.losses = nn.ModuleList(loss_modules)
        self.metrics = nn.ModuleList(metric_modules)
        self.transforms = nn.ModuleDict(transforms_dict)
        self.optim_factory = optim_factory
        self.lr_sched_factory = lr_sched_factory

    def forward(self, batch: TensorDict) -> TensorDict:
        return self.model(batch)

    def training_step(self, batch: TensorDict, batch_idx: int):
        batch = self(batch)
        batch = self.transforms["targets"](batch)

        loss_dict = {}
        loss = 0
        for loss_function in self.losses:
            out_key = loss_function.out_keys[0]
            _, name = out_key.split(".")
            value = loss_function(batch)[out_key]

            loss_dict[f"train/{name}"] = value
            loss += loss_function._weight * value

        self.log_dict(loss_dict)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: TensorDict, batch_idx: int):
        batch = self(batch)
        batch = self.transforms["targets"](batch)

        val_dict = {}
        for modules in [self.losses, self.metrics]:
            metric = 0
            for module in modules:
                out_key = module.out_keys[0]
                _, name = out_key.split(".")
                value = module(batch)[out_key]

                val_dict[f"val/{name}"] = value
                metric += module._weight * value

        self.log_dict(val_dict, batch_size=len(batch))
        self.log("val/loss", metric, prog_bar=True, batch_size=len(batch))

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int):
        batch = self(batch)
        batch = self.transforms["preds"](batch)

        return batch

    def configure_optimizers(self):
        optimizer = self.optim_factory(self.parameters())
        if self.lr_sched_factory is None:
            return optimizer

        lr_scheduler = self.lr_sched_factory(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
