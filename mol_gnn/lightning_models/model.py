from __future__ import annotations

from typing import Callable

import lightning as L
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT

from mol_gnn.conf import TARGET_KEY_PREFIX
from mol_gnn.types import LossConfig, LRSchedConfig, ModuleConfig


def is_target_key(key: str):
    """Is the input key :attr:`key` of the form ``targets.*``?"""
    return key.split(".")[0] == TARGET_KEY_PREFIX


class SimpleModel(L.LightningModule):
    """A :class:`SimpleModel` is a generic class for composing (mostly) arbitrary models.

    The general recipe consists of three configuration dictionaries that define the model, the loss,
    and the evaluation/testing metrics, respectively. The model configuration dictionary defines a

    Parameters
    ----------
    model_config : dict[str, ModelModuleConfig]
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

    loss_config : dict[str, LossModuleConfig]
        A mapping from a name to a dictionary with the keys:

        - ``weight``: a float for the term's weight in the total loss
        - ``module``: a callable that returns a single tensor
        - ``in_keys``: the input keys of the module

        .. note::
            Each term will be placed into the tensordict under the key `("loss.<NAME>")`

        The overall training loss is computed as the weighted sum of all terms. For more
        details on the ``in_keys`` key, see :attr:`model_config`.

    metric_config : dict[str, LossModuleConfig]
        A mapping from a name to a dictionary with the keys:

        - ``weight``: a float for the term's weight in the total validation loss
        - ``module``: a callable that returns a single tensor
        - ``in_keys``: the input keys of the module

        .. note::
            Each term will be placed into the tensordict under the key `("metric.<NAME>")`

        The overall validation loss is computed as the weighted sum of all loss term values. For
        details on the ``in_keys`` key, see :attr:`model_config`.
    """

    def __init__(
        self,
        modules: dict[str, ModuleConfig],
        losses: dict[str, LossConfig],
        metrics: dict[str, LossConfig],
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
            wrapped_module = TensorDictModule(
                loss_config["module"], loss_config["in_keys"], [("loss", name)]
            )
            wrapped_module = TensorDictModule(
                loss_config["module"], loss_config["in_keys"], [f"loss.{name}"]
            )
            wrapped_module._weight = loss_config["weight"]
            loss_modules.append(wrapped_module)
            selected_out_keys.update([k for k in loss_config["in_keys"] if not is_target_key(k)])
        metric_modules = []
        for name, metric_config in metrics.items():
            wrapped_module = TensorDictModule(
                metric_config["module"], metric_config["in_keys"], [f"metric.{name}"]
            )
            wrapped_module._weight = metric_config["weight"]
            metric_modules.append(wrapped_module)
            selected_out_keys.update([k for k in metric_config["in_keys"] if not is_target_key(k)])

        selected_out_keys = None if keep_all_output else list(selected_out_keys)

        self.model = TensorDictSequential(*model_modules, selected_out_keys=selected_out_keys)
        self.loss_functions = nn.ModuleList(loss_modules)
        self.metrics = nn.ModuleList(metric_modules)
        self.optim_factory = optim_factory
        self.lr_sched_factory = lr_sched_factory

    def forward(self, batch: TensorDict) -> Tensor:
        return self.model(batch)

    def training_step(self, batch: TensorDict, batch_idx: int = 0):
        batch = self(batch)

        loss_dict = {}
        loss = 0
        for loss_function in self.loss_functions:
            batch = loss_function(batch)
            out_key = loss_function.out_keys[0]
            _, name = out_key
            val = batch[out_key]

            loss_dict[f"train/{name}"] = val
            loss += loss_function._weight * val

        self.log_dict(loss_dict)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: TensorDict, batch_idx: int = 0):
        batch = self(batch)

        val_dict = {}
        for module_list in [self.loss_functions, self.metrics]:
            metric = 0
            for module in module_list:
                batch = module(batch)
                out_key = module.out_keys[0]
                _, name = out_key
                val = batch[out_key]

                val_dict[f"val/{name}"] = val
                metric += module._weight * val

        self.log_dict(val_dict)
        self.log("val/loss", metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optim_factory(self.parameters())
        if self.lr_sched_factory is None:
            return optimizer

        lr_scheduler = self.lr_sched_factory(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


"""    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)

        lr_sched = NoamLikeLRSched(
            opt,
            self.warmup_epochs,
            self.trainer.max_epochs,
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            self.init_lr,
            self.max_lr,
            self.final_lr,
        )
        lr_sched_config = {
            "scheduler": lr_sched,
            "interval": "step" if isinstance(lr_sched, NoamLR) else "batch",
        }

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

lr_sched_factory : Callable
"""
