from collections import defaultdict
from typing import Self

from jaxtyping import Float
from tensordict import TensorDict, tensorclass
import torch
from torch import Tensor

from mol_gnn.data.models.sample import Sample
from mol_gnn.data.models.graph import BatchedGraph


@tensorclass
class Batch:
    G: BatchedGraph
    Y: Float[Tensor, "b t"] | None
    extra_data: TensorDict
    # X_f: Float[Tensor, "b d_f"] | None
    # w: Float[Tensor, "b"]
    # lt_mask: Bool[Tensor, "b t"] | None
    # gt_mask: Bool[Tensor, "b t"] | None

    @classmethod
    def collate(cls, samples: list[Sample]) -> Self:
        Gs = [sample.pop("G") for sample in samples]

        G = BatchedGraph.from_graphs(Gs)
        if "y" in samples[0]:
            ys = [sample.pop("y") for sample in samples]
            Y = torch.as_tensor(ys)
        else:
            Y = None

        extra_data = defaultdict(list)
        for sample in samples:
            for key, value in sample.items():
                sample[key].append(value)

        for key, value in extra_data.items():
            extra_data[key] = torch.stack(value)

        extra_data = TensorDict(extra_data, batch_size=[len(samples)])
        return cls(G, Y, extra_data)
