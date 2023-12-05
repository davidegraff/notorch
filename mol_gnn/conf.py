"""Global configuration variables for mol_gnn"""
from mol_gnn.featurizers.molgraph.molecule import BaseMoleculeMolGraphFeaturizer


DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = BaseMoleculeMolGraphFeaturizer().shape
DEFAULT_MESSAGE_DIM = 300
DEFAULT_OUTPUT_DIM = DEFAULT_MESSAGE_DIM
