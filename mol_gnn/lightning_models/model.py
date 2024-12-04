from __future__ import annotations

import lightning as L
from torch import nn, Tensor, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from mol_gnn.schedulers import NoamLR, NoamLikeLRSched
from mol_gnn.types import ModelModuleConfig, LossModuleConfig


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
        A mapping from a name to a dictionary containing keys:

        - ``weight``: a float for the term's weight in the total loss
        - ``module``: a callable that returns a single tensor
        - ``in_keys``: the input keys of the module

        .. note::
            Each term will be placed into the tensordict under the nested key `("loss", KEY)`

        The overall training loss is computed as the weighted sum of all terms. For more
        details on the ``in_keys`` key, see :attr:`model_config`.

    metric_config : dict[str, LossModuleConfig]
        A mapping from a name to a dictionary containing keys:

        - ``weight``: a float for the term's weight in the total validation loss
        - ``module``: a callable that returns a single tensor
        - ``in_keys``: the input keys of the module

        .. note::
            Each term will be placed into the tensordict under the nested key `("metric", KEY)`

        The overall validation loss is computed as the weighted sum of all loss term values. For
        details on the ``in_keys`` key, see :attr:`model_config`.
    """

    def __init__(
        self,
        model_config: dict[str, ModelModuleConfig],
        loss_config: dict[str, LossModuleConfig],
        metric_config: dict[str, LossModuleConfig],
    ):
        super().__init__()

        modules = [
            TensorDictModule(module, in_keys, [(name, key) for key in out_keys])
            for name, (module, in_keys, out_keys) in model_config.items()
        ]

        selected_out_keys = []
        loss_modules = []
        for name, (weight, module, in_keys) in loss_config.items():
            wrapped_module = TensorDictModule(module, in_keys, [("loss", name)])
            wrapped_module._weight = weight
            loss_modules.append(wrapped_module)
            selected_out_keys += in_keys
        metric_modules = []
        for name, (weight, module, in_keys) in metric_config.items():
            wrapped_module = TensorDictModule(module, in_keys, [("metric", name)])
            wrapped_module._weight = weight
            metric_modules.append(wrapped_module)
            selected_out_keys += in_keys

        self.model = TensorDictSequential(modules, selected_out_keys=selected_out_keys)
        self.loss_functions = nn.ModuleList(loss_modules)
        self.metrics = nn.ModuleList(metric_modules)

    def forward(self, td: TensorDict) -> Tensor:
        return self.model(td)

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
