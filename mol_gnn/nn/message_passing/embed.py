from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn

from mol_gnn.conf import (
    DEFAULT_ATOM_DIM,
    DEFAULT_ATOM_HIDDEN,
    DEFAULT_BOND_DIM,
    DEFAULT_BOND_HIDDEN,
    DEFAULT_HIDDEN_DIM,
)
from mol_gnn.utils.registry import ClassRegistry


class InputEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()

        self.W = nn.Linear(input_dim, output_dim, bias)

    def forward(self, X: Tensor):
        return self.W(X)

    @classmethod
    def node(cls, bias: bool = True) -> InputEmbedding:
        return cls(DEFAULT_ATOM_DIM, DEFAULT_HIDDEN_DIM, bias)

    @classmethod
    def edge(cls, bias: bool = True) -> InputEmbedding:
        return cls(DEFAULT_ATOM_DIM + DEFAULT_BOND_DIM, DEFAULT_HIDDEN_DIM, bias)


class OutputEmbedding(nn.Module):
    output_dim: int

    @abstractmethod
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_HIDDEN_DIM,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        pass

    @abstractmethod
    def forward(self,V: Tensor, M: Tensor, V_d: Tensor):
        pass


OutputEmbeddingRegistry = ClassRegistry[OutputEmbedding]()


@OutputEmbeddingRegistry.register("linear")
class LinearOutputEmbedding(nn.Module):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_HIDDEN_DIM,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        block = nn.Sequential(
            nn.Linear(atom_dim + message_dim, message_dim, bias), nn.ReLU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(block)

    @property
    def output_dim(self) -> int:
        return self.blocks[-1][0].out_features

    def forward(self, V: Tensor, M: Tensor, V_d: Tensor):
        return self.blocks[0](torch.cat([V, M], dim=1))


@OutputEmbeddingRegistry.register("descriptors")
class AtomDescriptorEmbedding(LinearOutputEmbedding):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_HIDDEN_DIM,
        desc_dim: int = 0,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(atom_dim, bond_dim, message_dim, bias=bias, dropout=dropout, **kwargs)

        block = nn.Sequential(
            nn.Linear(message_dim + desc_dim, message_dim + desc_dim, bias), nn.Dropout(dropout)
        )
        self.blocks.append(block)

    def forward(self, V: Tensor, M: Tensor, V_d: Tensor):        
        try:
            H = self.blocks[0](torch.cat([V, M], dim=1))
            H = self.blocks[1](torch.cat([H, V_d], dim=1))
        except RuntimeError:
            raise ValueError(f"arg 'V_d' must be supplied when using {self.__class__.__name__}.")

        return H
