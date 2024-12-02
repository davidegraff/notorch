from torch import nn

def MLP(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 300,
    n_layers: int = 1,
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    r"""An :class:`MLP` is an FFN that implements the following function:

    .. math::
        \mathbf h_0 &= \mathbf x\,\mathbf W^{(0)} + \mathbf b^{(0)} \\
        \mathbf h_l &= \mathtt{dropout} \left(
            \tau \left(\,\mathbf h_{l-1}\,\mathbf W^{{l)} \right)
        \right) \\
        \mathbf h_L &= \mathbf h_{L-1} \mathbf W^{{l)} + \mathbf b^{{l)},

    where :math:`\mathbf x` is the input tensor, :math:`\mathbf W^{{l)}` is the learned weight
    matrix for the :math:`l`-th layer, :math:`\mathbf b^{{l)}` is the bias vector for the
    :math:`l`-th layer, :math:`\mathbf h^{{l)}` is the hidden representation at layer :math:`l`,
    :math:`\tau` is a nonlinearity (e.g., ReLU), and :math:`L` is the number of layers.
    """

    dropout = nn.Dropout(dropout)
    act = activation()

    dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
    blocks = [[dropout, nn.Linear(d1, d2), act] for d1, d2 in zip(dims[:-1], dims[1:])]
    layers = sum(blocks, [])

    return nn.Sequential(*layers[1:-1])
