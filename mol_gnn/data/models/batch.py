from typing import Iterable, NamedTuple, Self

from jaxtyping import Float, Bool
import torch
from torch import Tensor

from mol_gnn.data.models.datum import Datum
from mol_gnn.data.models.graph import BatchedGraph


class MpnnBatch(NamedTuple):
    G: BatchedGraph
    V_d: Float[Tensor, "V d_f"] | None
    X_f: Float[Tensor, "b d_f"] | None
    Y: Float[Tensor, "b d_o"] | None
    w: Float[Tensor, "b"]
    lt_mask: Bool[Tensor, "b d_o"] | None
    gt_mask: Bool[Tensor, "b d_o"] | None

    @classmethod
    def collate(cls, batch: Iterable[Datum]) -> Self:
        Gs, V_ds, x_fs, ys, weights, lt_masks, gt_masks = zip(*batch)

        return (
            BatchedGraph(Gs),
            None if V_ds[0] is None else torch.cat(V_ds, dim=0),
            None if x_fs[0] is None else torch.cat(x_fs, dim=0),
            None if ys[0] is None else torch.cat(ys),
            torch.tensor(weights).unsqueeze(1),
            None if lt_masks[0] is None else torch.cat(lt_masks),
            None if gt_masks[0] is None else torch.cat(gt_masks),
        )


class MultiInputMpnnBatch(NamedTuple):
    Gs: list[BatchedGraph]
    V_ds: list[Float[Tensor, "V d_h"] | None]
    X_f: Float[Tensor, "b d_z"] | None
    Y: Float[Tensor, "b d_o"] | None
    w: Float[Tensor, "b"]
    lt_mask: Bool[Tensor, "b d_o"] | None
    gt_mask: Bool[Tensor, "b d_o"] | None

    @classmethod
    def collate(cls, batches: Iterable[Iterable[Datum]]) -> Self:
        input_batches = [MpnnBatch.collate(batch) for batch in zip(*batches)]

        return MultiInputMpnnBatch(
            [batch.G for batch in input_batches],
            [batch.V_d for batch in input_batches],
            input_batches[0].X_f,
            input_batches[0].Y,
            input_batches[0].w,
            input_batches[0].lt_mask,
            input_batches[0].gt_mask,
        )