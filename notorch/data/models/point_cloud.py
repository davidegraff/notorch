from dataclasses import InitVar, dataclass, field
import textwrap
from typing import Iterable, Self

from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.types import Device

from notorch.conf import REPR_INDENT


@dataclass(repr=False, eq=False)
class PointCloud:
    R: Float[Tensor, "V d_r"]
    """a tensor of shape ``V x d_r`` containing the coordinates of each node in the point cloud."""
    t: Int[Tensor, "V t"]
    """a tensor of shape ``V x t`` containing the types of each node in the point cloud."""
    device_: InitVar[Device] = field(default=None, kw_only=True)

    def __post_init__(self, device_: Device):
        self.__device = device_
        self.to(device_)

    @property
    def num_nodes(self) -> int:
        return len(self.R)

    @property
    def device(self) -> Device:
        return self.__device

    def to(self, device: Device) -> Self:
        self.__device = device

        self.R = self.R.to(device)
        self.t = self.t.to(device)

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
            f"R: Tensor(shape={self.R.shape})",
            f"t: Tensor(shape={self.t.shape})",
            f"device={self.__device}",
            "",
        ]


@dataclass(repr=False, eq=False, kw_only=True)
class BatchedPointCloud(PointCloud):
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
    def from_point_clouds(cls, Ps: Iterable[PointCloud]):
        Rs = []
        ts = []
        batch_indices = []
        offset = 0

        for i, P in enumerate(Ps):
            Rs.append(P.R)
            ts.append(P.t)
            batch_indices.extend([i] * P.num_nodes)

            offset += P.num_nodes

        R = torch.cat(Rs, dim=0)
        t = torch.cat(ts, dim=0)
        batch_index = torch.tensor(batch_indices, dtype=torch.long)
        size = i + 1

        return cls(R, t, batch_index=batch_index, size=size, device_=P.device)

    def _build_field_info(self) -> list[str]:
        return [
            f"R: Tensor(shape={self.R.shape})",
            f"t: Tensor(shape={self.t.shape})",
            f"device={self.__device}",
            f"batch_size={len(self)}" "",
        ]
