from abc import abstractmethod

from torch import Tensor, nn

from mol_gnn.conf import DEFAULT_MESSAGE_DIM
from mol_gnn.utils.registry import ClassRegistry


class UpdateFunction(nn.Module):
    @abstractmethod
    def forward(self, M: Tensor, H_0: Tensor):
        """Calculate the updated hidden state for each edge"""


UpdateFunctionRegistry = ClassRegistry[UpdateFunction]()


@UpdateFunctionRegistry.register("linear")
class LinearUpdate(UpdateFunction):
    def __init__(
        self, message_dim: int = DEFAULT_MESSAGE_DIM, bias: bool = True, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.W_h = nn.Linear(message_dim, message_dim, bias)
        self.tau = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, M: Tensor, H_0: Tensor):
        H_t = self.W_h(M)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)

        return H_t
