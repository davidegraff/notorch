
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import textwrap
from typing import Iterable, Self

from jaxtyping import Float, Int, Num
from tensordict import tensorclass
import torch
from torch import Tensor
from torch.types import Device

from notorch.conf import REPR_INDENT


@tensorclass
class DualRankFeatures:
    scalar_feats: Num[Tensor, "*n d_s"]
    """a tensor of shape ``V x t`` containing the scalar node types/features."""
    vector_feats: Float[Tensor, "*n r d_v"]
    """a tensor of shape ``V x r x d_v`` containing the vector node features of each node."""


@dataclass(repr=False, eq=False)
class GVPPointCloud:
    node_feats: DualRankFeatures
    """the scalar and vector node features."""
    coords: Float[Tensor, "V r"]
    """a tensor of shape ``V x r`` containing the node coordinates."""
    device_: InitVar[Device] = field(default=None, kw_only=True)

    def __post_init__(self, device_: Device):
        self.__device = device_
        self.to(device_)

    @property
    def num_nodes(self) -> int:
        return len(self.coords)

    @property
    def device(self) -> Device:
        return self.__device

    def to(self, device: Device) -> Self:
        self.__device = device

        self.scalar_feats = self.scalar_feats.to(device)
        self.vector_feats = self.vector_feats.to(device)
        self.coords = self.coords.to(device)

        return self

    def __repr__(self) -> str:
        lines = (
            [f"{self.__class__.__name__}("]
            + [textwrap.indent(line, REPR_INDENT) for line in self._build_field_info()]
            + [")"]
        )

        return "\n".join(lines)

    def _build_field_info(self) -> list[str]:
        return [
            f"node_feats: {self.node_feats}",
            f"coords: Tensor(shape={self.coords.shape})",
            f"device={self.__device}",
            "",
        ]


@dataclass(repr=False, eq=False, kw_only=True)
class BatchedGVPPointCloud(GVPPointCloud):
    """A :class:`BatchedPointCloud` represents a batch of individual :class:`PointCloud`s."""

    batch_index: Int[Tensor, "V"]
    """A tensor of shape ``V`` containing the index of the parent :class:`PointCloud` of each node
    in the batched point cloud."""
    size: InitVar[int | None] = None
    """The number of point clouds in the batched input, if known. Otherwise, will be estimated via
    :code:`batch_index.max() + 1`"""

    def __post_init__(self, device_: torch.device | str | int | None, size: int | None):
        super().__post_init__(device_)

        self.__size = self.batch_index.max() + 1 if size is None else size

    def __len__(self) -> int:
        """The number of individual :class:`PointCloud`s in this batch"""
        return self.__size

    @classmethod
    def from_point_clouds(cls, Ps: Iterable[GVPPointCloud]):
        node_featss = []
        vector_featss = []
        coordss = []
        batch_indices = []
        offset = 0

        for i, P in enumerate(Ps):
            node_featss.append(P.node_feats)
            vector_featss.append(P.vector_feats)
            coordss.append(P.coords)
            batch_indices.extend([i] * P.num_nodes)

            offset += P.num_nodes

        node_feats = torch.cat(node_featss, dim=0)
        coords = torch.cat(coordss, dim=0)
        batch_index = torch.tensor(batch_indices, dtype=torch.long)
        size = i + 1

        return cls(
            node_feats, coords, batch_index=batch_index, size=size, device_=P.device
        )

    def _build_field_info(self) -> list[str]:
        return [
            f"node_feats: {self.node_feats}",
            f"coords: Tensor(shape={self.coords.shape})",
            f"batch_size={len(self)}",
            f"device={self.__device}",
            "",
        ]
