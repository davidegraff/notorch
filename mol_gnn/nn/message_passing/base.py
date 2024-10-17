from abc import abstractmethod

from torch import nn, Tensor
from mol_gnn.data.models.graph import BatchedGraph

from mol_gnn.utils.hparams import HasHParams


MessagePassingInput = tuple[BatchedGraph, Tensor | None]


class MessagePassing(nn.Module, HasHParams):
    """A :class:`MessagePassing` module encodes a batch of molecular graphs using message passing
    to learn vertex-level hidden representations."""
    output_dim: int

    @abstractmethod
    def forward(self, G: BatchedGraph, V_d: Tensor | None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        V : Tensor
            a tensor of shape ``V x d_v`` containing the vertex feature matix
        E : Tensor
            a tensor of shape ``E x d_e`` containing the edge feature matrix
        edge_index : Tensor
            a tensor of shape ``2 x E`` containing the adajency matrix in COO
            format
        rev_index : Tensor | None
            a tensor of shape ``E`` containing the reverse indices for each directed edge in the
            graph. If ``None``, this will be computed on-the-fly
        V_d : Tensor | None, default=None
            an optional tensor of shape ``V x d_vd`` containing additional descriptors for each
            atom in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape ``b x d_o``, where ``d_o`` is equal to :attr:`self.output_dim`,
            containing the encoding of each vertex in the batch
        """
