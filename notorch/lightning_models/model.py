from __future__ import annotations

from collections.abc import Callable

import lightning as L
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT

from notorch.conf import TARGET_KEY_PREFIX
from notorch.types import GroupTransformConfig, LossConfig, ModuleConfig

EPS = 1e-6


def is_target_key(key: str):
    """Is the input key :attr:`key` of the form ``targets.*``?"""
    return key.split(".")[0] == TARGET_KEY_PREFIX


class NotorchModel(L.LightningModule):
    """A :class:`NotorchModel` is a generic class for composing (mostly) arbitrary models.

    The general recipe consists of three configuration dictionaries that define the model, the loss,
    and the evaluation/testing metrics, respectively.

    Parameters
    ----------
    modules : dict[str, ModelModuleConfig]
        A mapping from a name to a dictionary with keys:

        - ``module``: the :class:`~torch.nn.Module` that will be wrapped inside a
        :class:`~tensordict.nn.TensorDictModule`.

        - ``in_keys``: the keys to retrieve from the tensordict whose values will be passed to the
        module as either

            - a list of keys that will be passed in as positional arguments

            - a dictionary mapping from keyword argument name to the key that will be
            fetched and supplied to the corresponding argument

        - ``out_keys``: the keys under which the module's output will be placed into the tensordict

        .. note::
            The output values will be placed in a sub-tensordict under the module's name (i.e.,
            the key corresponding to the 3-tuple)

    losses : dict[str, LossModuleConfig]
        A mapping from a loss term name to a dictionary with keys:

        - ``module``: any callable that returns a single scalar corresponding to a loss term
        - ``in_keys``: the values to retrieve from the tensordict whose values will be passed to the
        module

        Each individual term will be placed into the tensordict under the key ``losses.<NAME>``
        and will be logged to ``<train|val>/<NAME>``, depending on the whether the model is in
        training or validation mode.

        .. note::
            All loss terms will also be calculated in the validation step, so they *not* be
            defined again in :attr:`metrics`.

        .. important::
            The overall training loss will be logged to ``train/loss`` during training, so if you
            name any individual term simply ``loss``, then it it will be overwritten in the logs.
            It is best to give each term a descriptive name, such as ``mse`` for an MSE term.

        .. seealso::
            :paramref:`.modules`
                Details on the ``in_keys`` key

            :paramref:`train_loss_weights`
                Details on the how the overall training loss is computed.

    metrics : dict[str, LossModuleConfig]
        A mapping from a metric term name to a dictionary with keys:

        - ``module``: any callable that returns a single scalar corresponding to a metric term
        - ``in_keys``: the values to retrieve from the tensordict whose values will be passed to the
        module

        Each invidiual term will be placed into the tensordict under the key `metrics.<NAME>` and
        will be logged to ``val/<NAME>``

        .. seealso::
            :paramref:`.modules` for details on the ``in_keys`` key

            :paramref:`.val_loss_weights`
            for details on the how the overall validation loss is computed.


    transforms : dict[str, GroupTransformConfig] | None, default=None
        A mapping from a name to a nested dictionary that defines the configuration for both
        prediction and target transformation. The nested dictionary contains two keys, ``preds``
        and ``targets``, each corresponding to an inner dictionary with the following keys:

        - ``module``: any callable that has **no learnable parameters**

        - ``key``: the key in the tensordict whose value will be modified *in place* by the above
        callable.

        The ``preds`` transforms will be applied to model predictions at inference time via
        :meth:`predict_step` and the ``"targets"`` transforms will be applied to the
        input targets during training and validation.

        .. note::
            In the event that the specified keys are not present in the tensordict, then the
            transforms will have no effect. As such, you must take care to ensure the keys have been
            named correctly.

    train_loss_weights : dict[str, float] | None, default=None
        a mapping from loss term name to its weight in the overall training loss. If ``None``, then
        each term will be given a weight of ``1.0``.

    val_loss_weights : dict[str, float] | None, default=None
        a mapping from loss term name to its weight in the overall validation loss. If ``None``,
        then use the weights of :paramref:`.train_loss_weights`. As mentioned in
        :paramref:`.losses`, all training loss terms will be calculated in the validation step, so
        these terms may be included in the overall validation loss.

    optim_factory : Callable[[ParamsT], Optimizer], default=lambda params: Adam(params, lr=1e-4)
        a callable that takes the model's paramters and returns a :class:`~torch.optim.Optimizer`.
        By default, will return a :class:`torch.optim.Adam` with a learning rate of `1e-4`.

    lr_sched_factory : Callable[[Optimizer], LRScheduler | LRSchedulerConfigType] | None,
    default=None
        an optional callable that accepts in an optimizer and returns a
        :class:`~torch.optim.lr_scheduler.LRScheduler` or a dictionary configuring the learning rate
        scheduler.

        .. seealso::
            :meth:`LightningModule.configure_optimizers` for details on the structure of the
            dictionary.

    keep_all_output: bool, default=False
        If ``True``, retain all intermediate tensors in the output tensordict. Otherwise, keep only
        those required to calculate losses and metrics
    """

    def __init__(
        self,
        modules: dict[str, ModuleConfig],
        losses: dict[str, LossConfig],
        metrics: dict[str, LossConfig],
        transforms: dict[str, GroupTransformConfig] | None = None,
        train_loss_weights: dict[str, float] | None = None,
        val_loss_weights: dict[str, float] | None = None,
        optim_factory: Callable[[ParamsT], Optimizer] = lambda params: Adam(params, lr=1e-4),
        lr_sched_factory: Callable[[Optimizer], LRScheduler | LRSchedulerConfigType] | None = None,
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
        loss_modules = {}
        for name, loss_config in losses.items():
            module = TensorDictModule(
                loss_config["module"], loss_config["in_keys"], [f"losses.{name}"], inplace=False
            )
            # module._weight = loss_config["weight"]
            loss_modules[name] = module
            selected_out_keys.update([k for k in loss_config["in_keys"] if not is_target_key(k)])
        metric_modules = {}
        for name, metric_config in metrics.items():
            module = TensorDictModule(
                metric_config["module"],
                metric_config["in_keys"],
                [f"metrics.{name}"],
                inplace=False,
            )
            # module._weight = metric_config["weight"]
            metric_modules[name] = module
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

        if train_loss_weights is None:
            train_loss_weights = {name: 1.0 for name in loss_modules}
        if val_loss_weights is None:
            val_loss_weights = train_loss_weights

        self.model = TensorDictSequential(*model_modules, selected_out_keys=selected_out_keys)
        self.losses = nn.ModuleDict(loss_modules)
        self.metrics = nn.ModuleDict(metric_modules)
        self.transforms = nn.ModuleDict(transforms_dict)
        self.train_loss_weights = train_loss_weights
        self.val_loss_weights = val_loss_weights
        self.optim_factory = optim_factory
        self.lr_sched_factory = lr_sched_factory

    def forward(self, batch: TensorDict) -> TensorDict:
        return self.model(batch)

    def training_step(self, batch: TensorDict, batch_idx: int):
        batch = self(batch)
        batch = self.transforms["targets"](batch)

        loss_dict = {}
        train_loss = 0
        for name, loss_function in self.losses.items():
            out_key = loss_function.out_keys[0]
            value = loss_function(batch)[out_key]

            loss_dict[f"train/{name}"] = value
            train_loss += self.train_loss_weights.get(name, EPS) * value
            # train_loss += loss_function._weight * value

        self.log_dict(loss_dict)
        self.log("train/loss", train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch: TensorDict, batch_idx: int):
        batch = self(batch)
        batch = self.transforms["targets"](batch)

        loss_dict = {}
        val_loss = 0
        for name, module in self.losses.items():
            out_key = module.out_keys[0]
            value = module(batch)[out_key]

            loss_dict[f"val/{name}"] = value
            val_loss += self.val_loss_weights.get(name, EPS) * value

        for name, module in self.metrics.items():
            out_key = module.out_keys[0]
            # _, name = out_key.split(".")
            value = module(batch)[out_key]

            loss_dict[f"val/{name}"] = value
            val_loss += self.val_loss_weights.get(name, EPS) * value

        self.log_dict(loss_dict, batch_size=len(batch))
        self.log("val/loss", val_loss, prog_bar=True, batch_size=len(batch))

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

    def extra_repr(self) -> str:
        lines = [
            f"(train_loss_weights): {self.train_loss_weights}",
            f"(val_loss_weights): {self.val_loss_weights}",
        ]

        return "\n".join(lines)
