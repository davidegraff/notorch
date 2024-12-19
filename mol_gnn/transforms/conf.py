from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType, HybridizationType

# atom feature types
ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3]
HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
]
NUM_HS = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-1, -2, 1, 2, 0]

# bond feature types
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_STEREOS = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
    BondStereo.STEREOATROPCW,
]
