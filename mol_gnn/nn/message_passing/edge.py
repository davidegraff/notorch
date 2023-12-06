import torch
from torch import Tensor, nn

from mol_gnn.nn.agg import Aggregation, DirectedEdgeAggregation, EdgeAggregation, NodeAggregation
from mol_gnn.nn.message_passing.agg import DirectedEdgeAggregation, EdgeAggregation, NodeAggregation
from mol_gnn.nn.message_passing.embed import InputEmbedding, OutputEmbedding
from mol_gnn.nn.message_passing.message import MessageFunction
from mol_gnn.nn.message_passing.base import MessagePassing
from mol_gnn.nn.message_passing.update import UpdateFunction


def calc_rev_index(edge_index: Tensor) -> Tensor:
    edge_index = edge_index.T
    rev_mask = (edge_index[None, :] == edge_index.flip(-1)[:, None]).all(-1)

    return torch.where(rev_mask)[1]

    
class EdgeMessagePassing(MessagePassing):
    def __init__(
        self,
        embed: InputEmbedding | None,
        message: MessageFunction,
        agg: Aggregation,
        update: UpdateFunction,
        out_embed: OutputEmbedding,
        act: nn.Module,
        depth: int,
        directed: bool,
    ):
        super().__init__()

        self.embed = embed or InputEmbedding.edge()
        self.message = message
        self.edge_agg = DirectedEdgeAggregation(agg) if directed else EdgeAggregation(agg)
        self.update = update
        self.node_agg = NodeAggregation(agg)
        self.out_embed = out_embed
        self.act = act
        self.depth = depth

    @property
    def output_dim(self) -> int:
        return self.out_embed.output_dim

    def forward(
        self,
        V: Tensor,
        E: Tensor,
        edge_index: Tensor,
        rev_index: Tensor | None,
        V_d: Tensor | None,
    ) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        V : Tensor
            a tensor of shape ``V x d_v`` containing the node feature matix
        E : Tensor
            a tensor of shape ``E x d_e`` containing the edge feature matrix
        edge_index : Tensor
            a tensor of shape ``2 x E`` containing the adajency matrix in COO format
        rev_index : Tensor | None
            a tensor of shape ``E`` containing the reverse indices for each directed edge in the
            graph. If ``None``, this will be computed on-the-fly
        V_d : Tensor | None
            an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
            in the batch to be passed to the output embedding.

        Returns
        -------
        Tensor
            a tensor of shape ``b x d_o``, where ``d_o`` is equal to :attr:`self.output_dim`,
            containing the encoding of each vertex in the batch
        """
        src, dest = edge_index
        rev_index = calc_rev_index(edge_index) if rev_index is None else rev_index
        dim_size = len(V)

        H_0 = self.embed(torch.cat([V[src], E], dim=1))

        H = self.act(H_0)
        for _ in range(1, self.depth):
            M = self.message(H, V[src], E)
            M = self.edge_agg(M, edge_index, rev_index, dim_size)
            H = self.update(H, M, H_0)
        H_v = self.node_agg(H, edge_index, rev_index, dim_size)

        return self.out_embed(H_v, V, V_d)
