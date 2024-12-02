from abc import abstractmethod

from torch import Tensor, nn

from mol_gnn.conf import DEFAULT_HIDDEN_DIM
from mol_gnn.utils.registry import ClassRegistry


class UpdateFunction(nn.Module):
    @abstractmethod
    def forward(self, H: Tensor, M: Tensor, H_0: Tensor):
        """Calculate the updated hidden state for each state"""


UpdateFunctionRegistry = ClassRegistry[UpdateFunction]()


@UpdateFunctionRegistry.register("residual")
class ResidualUpdate(UpdateFunction):
    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim

        self.W = nn.Linear(input_dim, output_dim, bias)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: Tensor, M: Tensor, H_0: Tensor):
        H = self.W(M)
        H = self.act(H_0 + H)
        H = self.dropout(H)

        return H
