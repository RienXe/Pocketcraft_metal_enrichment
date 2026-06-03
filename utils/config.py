"""Module containing global configuration values and constants."""

import json
import os
from rdkit.Chem.rdchem import BondType as BT

# 1. Feature Configuration Defaults
FEATURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "protein_feature_defaultv2.json")
DEFAULT_LIGAND_FEATURE_JSON = os.path.join(os.path.dirname(__file__), "..", "ligand_feature_combined_refined.json")

# RCSB endpoints for ligand templates
RCSB_CIF_URL = "https://files.rcsb.org/ligands/download/{ccd}.cif"
RCSB_SDF_URL = "https://files.rcsb.org/ligands/download/{ccd}_ideal.sdf"

# 9D atom categorical feature vocab
VOCAB = {
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": list(range(0, 11)) + ["misc"],
    "formal_charge": list(range(-5, 6)) + ["misc"],
    "num_Hs": list(range(0, 9)) + ["misc"],
    "num_radical_e": list(range(0, 5)) + ["misc"],
    "hybridization": [
        "UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER",
    ],
    "is_aromatic": [False, True],
    "num_rings": list(range(0, 7)) + ["misc"],
}

BOND_TYPE_MAP = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
}

METAL_ATOMIC_NUMS = {
    "ZN": 30, "CA": 20, "MG": 12,
    "CU": 29, "CU1": 29,
    "FE": 26, "FE2": 26,
    "MN": 25, "CO": 27, "NI": 28,
    'NA': 11, 'K': 19
}

METAL_VOCABULARY = ['UNK', 'ZN', 'CA', 'MG', 'FE', 'MN', 'CU', 'NI', 'CO', 'NA', 'K']
METAL_TO_IDX = {m: i for i, m in enumerate(METAL_VOCABULARY)}

def load_config(path):
    """
    Load a JSON configuration file.

    Args:
        path (str): Path to the JSON configuration file.

    Returns:
        dict: The parsed JSON data.
    """
    with open(path, "r") as f:
        return json.load(f)
