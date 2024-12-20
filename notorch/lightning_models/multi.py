from typing import Iterable

import torch
from torch import Tensor

from notorch.data.models.graph import BatchedGraph
from notorch.lightning_models.mpnn import MPNN
from notorch.nn import Aggregation, MultiInputMessagePassing, Predictor
from notorch.nn.metrics import Metric


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MultiInputMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = True,
        metrics: Iterable[Metric] | None = None,
        w_t: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            w_t,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
        )
        self.message_passing: MultiInputMessagePassing

    def fingerprint(
        self, bmgs: Iterable[BatchedGraph], V_ds: Iterable[Tensor], X_f: Tensor | None = None
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.graph_node_index) for H_v, bmg in zip(H_vs, bmgs)]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        return H if X_f is None else torch.cat((H, X_f), 1)
