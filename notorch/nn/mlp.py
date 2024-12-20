from collections.abc import Sequence
from math import prod

import torch.nn as nn

from notorch.conf import DEFAULT_HIDDEN_DIM


def MLP(
    input_dim: int,
    output_size: int | Sequence[int],
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_layers: int = 1,
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    r"""An :class:`MLP` is a :class:`~torch.nn.Module` that implements the following function:

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

    Parameters
    ----------
    input_dim : int
        the input dimension
    output_size : int | Sequence[int]
        the size of the output dimension as either a single integer or the unflattened shape. If the
        latter is provided, then the final dimension will be unflattened to the desired shape.
    hidden_dim : int, default=DEFAULT_HIDDEN_DIM
        _description_
    num_layers : int, default=1
        the number of hidden layers
    dropout : float, default=0.0
        the dropout probability.
    activation : type[nn.Module], default=nn.ReLU
        the nonlinearity.

    Returns
    -------
    nn.Sequential
        the MLP
    """
    if isinstance(output_size, int):
        output_dim = output_size
        unflatten = None
    else:
        output_dim = prod(output_size)
        unflatten = nn.Unflatten(-1, output_size)
    drop = nn.Dropout(dropout)
    act = activation()

    dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
    blocks = [[drop, nn.Linear(d1, d2), act] for d1, d2 in zip(dims[:-1], dims[1:])]
    layers = sum(blocks, [])
    mlp = nn.Sequential(*layers[1:-1])
    if unflatten is not None:
        mlp.append(unflatten)

    return mlp
