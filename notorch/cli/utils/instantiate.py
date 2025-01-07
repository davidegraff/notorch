"""largely copied from https://github.com/ashleve/lightning-hydra-template/tree/main"""
import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


def instantiate_callbacks(cfg: DictConfig) -> list[Callback] | None:
    callback_config = hydra.utils.instantiate(cfg)
    if callback_config is None:
        return None

    callbacks = []
    for name, callback in callback_config.items():
        if not isinstance(callback, Callback):
            raise ValueError(
                "invalid callback configured! "
                f"Callback under name '{name}' is not of type `Callback`!"
            )
        callbacks.append(callback)

    return callbacks


def instantiate_loggers(cfg: DictConfig) -> list[Logger] | None:
    logger_config = hydra.utils.instantiate(cfg)
    if logger_config is None:
        return None

    loggers = []
    for name, logger in logger_config.items():
        if not isinstance(logger, Logger):
            raise ValueError(
                f"invalid logger configured! Logger under name '{name}' is not of type `Logger`!"
            )
        loggers.append(logger)

    return loggers


callbacks = instantiate_callbacks
loggers = instantiate_loggers
