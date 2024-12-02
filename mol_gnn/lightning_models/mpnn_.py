from __future__ import annotations

from typing import Iterable

import lightning as L
import torch
from torch import nn, Tensor, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from mol_gnn.data.models.graph import BatchedGraph
from mol_gnn.data.models.batch import MpnnBatch
from mol_gnn.nn.gnn.base import GNNLayer, Aggregation
from mol_gnn.schedulers import NoamLR


class MPNN(L.LightningModule):
    def __init__(
        self,
        encoder: GNNLayer,
        agg: Aggregation,
        predictor: nn.Module,
        loss: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor

    def fingerprint(
        self, G: BatchedGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """The learned fingerprints for the input molecules"""
        H = self.encoder(G, V_d, len(G))
        H = self.bn(H)

        return H if X_f is None else torch.cat((H, X_f), 1)

    def encoding(
        self, G: BatchedGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """The final hidden representations for the input molecules"""
        return self.predictor[:-1](self.fingerprint(G, V_d, X_f))

    def forward(
        self, G: BatchedGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.predictor(self.fingerprint(G, V_d, X_f))

    def training_step(self, batch: MpnnBatch, batch_idx):
        G, V_d, X_f, targets, w_s, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(G, V_d, X_f)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, w_s, self.task_weights, lt_mask, gt_mask)

        self.log("train/loss", l, prog_bar=True)

        return l

    def validation_step(self, batch: MpnnBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))
        self.log("val_loss", losses[0], batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: MpnnBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"test/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))

    def _evaluate_batch(self, batch) -> list[Tensor]:
        G, V_d, X_f, targets, _, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(G, V_d, X_f)

        return [
            metric(preds, targets, mask, None, None, lt_mask, gt_mask)
            for metric in self.metrics[:-1]
        ]

    def predict_step(self, batch: MpnnBatch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : TrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
            molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
            per task. The final dimension is flattened, so that the targets for each task are
            grouped. I.e., the first ``t`` elements are the first target for each task, the second
            ``t`` elements the second target, etc.
            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        G, X_vd, X_f, *_ = batch

        return self(G, X_vd, X_f)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)

        lr_sched = NoamLR(
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

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        hparams = torch.load(checkpoint_path)["hyper_parameters"]

        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
        }

        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )
