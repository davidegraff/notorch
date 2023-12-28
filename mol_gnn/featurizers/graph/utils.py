import numpy as np

from mol_gnn.featurizers.graph.graph import Graph


def find_unique_edge_indices(edge_index: np.ndarray):
    """thanks to ChatGPT for this function"""
    sorted_edge_index = np.sort(edge_index, axis=0)
    _, indices = np.unique(sorted_edge_index, axis=1, return_index=True)

    return indices


def line_graph(mg: Graph):
    V, E, edge_index, _ = mg
    src, dest = edge_index

    V_line_bidir = np.concatenate([(V[src] + V[dest]) / 2, E], axis=1)
    uniq_idxs = find_unique_edge_indices(edge_index)
    V_line = V_line_bidir[uniq_idxs]

def line_graph(mg: Graph):
    """Return the line graph of a molecular graph"""
    V = mg.V
    E = mg.E
    edge_index = mg.edge_index
    rev_index = mg.rev_index

    n_atoms = V.shape[0]
    n_bonds = E.shape[0]

    V_line = np.empty((2 * n_bonds, V.shape[1]))
    E_line = np.empty((2 * n_bonds, E.shape[1]))
    edge_index_line = [[], []]

    for i in range(n_bonds):
        u, v = edge_index[:, i]
        u_line, v_line = rev_index[2 * i : 2 * i + 2]

        V_line[u_line] = V[u]
        V_line[v_line] = V[v]

        E_line[u_line] = E[i]
        E_line[v_line] = E[i]

        edge_index_line[0].extend([u_line, v_line])
        edge_index_line[1].extend([v_line, u_line])

    edge_index_line = np.array(edge_index_line, int)

    return Graph(V_line, E_line, edge_index_line, rev_index)


def line_graph(mg):
    V, E, (src, dest), rev_index = mg
    n_atoms = V.shape[0]
    n_bonds = E.shape[0]

    V_line = np.concatenate([V[src, E]], axis=1)
    E_line = np.concatenate([E, E], axis=0)