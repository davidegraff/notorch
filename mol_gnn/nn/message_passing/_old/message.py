from abc import abstractmethod

import torch
from torch import Tensor, nn

from mol_gnn.utils.registry import ClassRegistry


class MessageFunction(nn.Module):
    @abstractmethod
    def forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        """Calculate the message for each edge given its current hidden state"""
    
MessageFunctionRegistry = ClassRegistry[MessageFunction]()


@MessageFunctionRegistry.register("identity")
class Identity(MessageFunction):
    def forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        return H


@MessageFunctionRegistry.register("cat-edge")
class CatEdge(MessageFunction):
    def forward(self, H: Tensor, V: Tensor, E: Tensor) -> Tensor:
        return torch.cat([H, E], dim=1)
