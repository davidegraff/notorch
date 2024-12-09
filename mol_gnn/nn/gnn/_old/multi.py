from typing import Iterable

import torch.nn as nn

from mol_gnn.nn.message_passing.base import MessagePassing, MessagePassingInput


class MultiInputMessagePassing(nn.Module):
    """A :class:`MulticomponentMessagePassing` performs message-passing on each individual input in
    a multicomponent input then concatenates the representation of each input to construct a
    global representation
    """

    convs: list[MessagePassing]

    def __init__(self, convs: Iterable[MessagePassing]):
        super().__init__()

        self.convs = nn.ModuleList(convs)
        if len(self.convs) == 0:
            raise ValueError("arg 'convs' was empty!")

    def __len__(self) -> int:
        return len(self.convs)

    @property
    def shared(self) -> bool:
        """Whether the same message passing block is shared among all components"""
        return all([b_i is b_j for b_i, b_j in zip(self.convs[:-1], self.convs[1:])])

    @property
    def output_dim(self) -> int:
        return sum(conv.output_dim for conv in self.convs)

    def forward(self, inputs: Iterable[MessagePassingInput]) -> list[Tensor]:
        """Encode the multicomponent inputs

        Parameters
        ----------
        bmgs : Iterable[MessagePassingInput]

        Returns
        -------
        list[Tensor]
            a list of tensors of shape `b x d_o^k` containing the respective encodings of the
            :math:`k`-th component, where ``b`` is the batch size and ``d_o^k`` is the output
            dimension of the :math:`k`-th encoder
        """
        return [block(G, V_d) for block, (G, V_d) in zip(self.convs, inputs)]

    @classmethod
    def shared(cls, conv: MessagePassing, n_components: int):
        """
        Parameters
        ----------
        conv : MessagePassing
            the message-passing block to share among all components
        n_components : int
            the number of components in each input

        Returns
        -------
        MultiInputMessagePassing
        """
        return cls([conv] * n_components)
