from abc import abstractmethod

import torch
from torch import Tensor, nn

from mol_gnn.conf import DEFAULT_ATOM_DIM, DEFAULT_BOND_DIM, DEFAULT_MESSAGE_DIM
from mol_gnn.utils.registry import ClassRegistry


class InputEmbedding(nn.Module):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_MESSAGE_DIM,
        bias: bool = True
    ):
        super().__init__()

    @abstractmethod
    def forward(self, V: Tensor, E: Tensor, edge_index: Tensor) -> Tensor:
        pass


InputEmbeddingRegistry = ClassRegistry[InputEmbedding]()


@InputEmbeddingRegistry.register("bond")
class BondMessageEmbedding(InputEmbedding):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_MESSAGE_DIM,
        bias: bool = True
    ):
        super().__init__()

        self.W_i = nn.Linear(atom_dim + bond_dim, message_dim, bias)

    def forward(self, V: Tensor, E: Tensor, edge_index: Tensor):
        return self.W_i(torch.cat([V[edge_index[0]], E], dim=1))


@InputEmbeddingRegistry.register("atom")
class AtomMessageEmbedding(InputEmbedding):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_MESSAGE_DIM,
        bias: bool = True
    ):
        super().__init__()

        self.W_i = nn.Linear(atom_dim, message_dim, bias)

    def forward(self, V: Tensor, E: Tensor, edge_index: Tensor):
        return self.W_i(V[edge_index[0]])


class OutputEmbedding(nn.Module):
    output_dim: int

    @abstractmethod
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_MESSAGE_DIM,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        pass

    @abstractmethod
    def forward(self, M: Tensor, V: Tensor, V_d: Tensor):
        pass


OutputEmbeddingRegistry = ClassRegistry[OutputEmbedding]()


@OutputEmbeddingRegistry.register("linear")
class LinearOutputEmbedding(nn.Module):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_MESSAGE_DIM,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        block = nn.Sequential(
            nn.Linear(atom_dim + message_dim, message_dim, bias),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(block)
    
    @property
    def output_dim(self) -> int:
        return self.blocks[-1][0].out_features
    
    def forward(self, M: Tensor, V: Tensor, V_d: Tensor):
        return self.blocks[0](torch.cat([V, M], dim=1))


@OutputEmbeddingRegistry.register("descriptors")
class AtomDescriptorEmbedding(LinearOutputEmbedding):
    def __init__(
        self,
        atom_dim: int = DEFAULT_ATOM_DIM,
        bond_dim: int = DEFAULT_BOND_DIM,
        message_dim: int = DEFAULT_MESSAGE_DIM,
        desc_dim: int = 0,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(
            atom_dim, bond_dim, message_dim, bias=bias, dropout=dropout, **kwargs
        )

        block = nn.Sequential(
            nn.Linear(message_dim + desc_dim, message_dim + desc_dim, bias),
            nn.Dropout(dropout)
        )
        self.blocks.append(block)
        
    def forward(self, M: Tensor, V: Tensor, V_d: Tensor):
        if V_d is None:
            raise ValueError(f"arg 'V_d' must be supplied when using {self.__class__.__name__}.")
        
        H = self.blocks[0](torch.cat([V, M], dim=1))
        H = self.blocks[1](torch.cat([H, V_d], dim=1))

        return H
