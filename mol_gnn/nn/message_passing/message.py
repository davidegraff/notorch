from abc import abstractmethod

import torch
from torch import Tensor, nn

from mol_gnn.utils.registry import ClassRegistry


class MessageFunction(nn.Module):
    def __init__(self, directed: bool = True) -> None:
        super().__init__()

        self.directed = directed

    def forward(self, H: Tensor, V: Tensor, E: Tensor, rev_index: Tensor) -> Tensor:
        """Calculate the message for each edge given its current hidden state"""
        H = (H + H[rev_index]) / 2 if not self.directed else H

        return self._forward(H, V, E)
    
    @abstractmethod
    def _forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        pass

MessageFunctionRegistry = ClassRegistry[MessageFunction]()


@MessageFunctionRegistry.register("identity")
class Identity(MessageFunction):
    def _forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        return H


@MessageFunctionRegistry.register("atom")
class AtomMessages(MessageFunction):
    def _forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        return torch.cat([H, E], dim=1)
