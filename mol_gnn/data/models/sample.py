from typing import TypedDict

from mol_gnn.data.models.graph import Graph


class Sample(TypedDict, total=False):
    G: Graph
