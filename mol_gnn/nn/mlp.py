from torch import nn


class MLP(nn.Sequential):
    r"""An :class:`MLP` is an FFN that implements the following function:

    .. math::
        \mathbf h_0 &= \mathbf x\,\mathbf W^{(0)} + \mathbf b^{(0)} \\
        \mathbf h_l &= \mathtt{dropout} \left(
            \sigma \left(\,\mathbf h_{l-1}\,\mathbf W^{{l)} \right)
        \right) \\
        \mathbf h_L &= \mathbf h_{L-1} \mathbf W^{{l)} + \mathbf b^{{l)},

    where :math:`\mathbf x` is the input tensor, :math:`\mathbf W^{{l)}` is the learned weight
    matrix for the :math:`l`-th layer, :math:`\mathbf b^{{l)}` is the bias vector for the
    :math:`l`-th layer, :math:`\mathbf h^{{l)}` is the hidden representation at layer :math:`l`,
    :math:`\sigma` is the activation function, and :math:`L` is the number of layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU()
    ):
        super().__init__()

        dropout = nn.Dropout(dropout)
        act = activation()

        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        blocks = [[dropout, nn.Linear(d1, d2), act] for d1, d2 in zip(dims[:-1], dims[1:])]
        layers = sum(blocks, [])

        super().__init__(*layers[1:-1])
