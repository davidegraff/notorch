from abc import abstractmethod

import torch
from torch import Tensor, nn

from mol_gnn.utils.registry import ClassRegistry


class MessageFunction(nn.Module):
    def __init__(self, directed: bool = True) -> None:
        super().__init__()

        self.directed = directed

    @abstractmethod
    def forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        """Calculate the message for each edge given its current hidden state"""
    
MessageFunctionRegistry = ClassRegistry[MessageFunction]()


@MessageFunctionRegistry.register("identity")
class Identity(MessageFunction):
    def forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        return H


@MessageFunctionRegistry.register("node")
class NodeMessages(MessageFunction):
    def forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        return torch.cat([H, E], dim=1)
