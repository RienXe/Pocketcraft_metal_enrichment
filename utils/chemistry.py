"""Module for chemical and molecular graph operations using RDKit."""

import requests
import torch
import torch.nn.functional as F
from rdkit import Chem
from utils.config import VOCAB, BOND_TYPE_MAP, RCSB_CIF_URL, RCSB_SDF_URL

def safe_idx(values, val):
    """
    Safely find the index of a value in a list, falling back to 'misc' or 0.

    Args:
        values (list): List of possible vocabulary values.
        val (any): The value to find.

    Returns:
        int: The index of the value or the fallback index.
    """
    try:
        return values.index(val)
    except ValueError:
        return values.index("misc") if "misc" in values else 0

def atom_feature_vec(atom: Chem.Atom, ring_info) -> torch.Tensor:
    """
    Generate a 9-dimensional discrete feature vector for an RDKit atom.

    Args:
        atom (Chem.Atom): The input atom.
        ring_info (rdkit.Chem.RingInfo): Ring information for the molecule.

    Returns:
        torch.Tensor: Feature vector of shape [9].
    """
    chiral = str(atom.GetChiralTag())
    if chiral in {"CHI_SQUAREPLANAR", "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL"}:
        chiral = "CHI_OTHER"
    return torch.tensor(
        [
            safe_idx(VOCAB["atomic_num"], atom.GetAtomicNum()),
            safe_idx(VOCAB["chirality"], chiral),
            safe_idx(VOCAB["degree"], atom.GetTotalDegree()),
            safe_idx(VOCAB["formal_charge"], atom.GetFormalCharge()),
            safe_idx(VOCAB["num_Hs"], atom.GetTotalNumHs()),
            safe_idx(VOCAB["num_radical_e"], atom.GetNumRadicalElectrons()),
            safe_idx(VOCAB["hybridization"], str(atom.GetHybridization())),
            VOCAB["is_aromatic"].index(atom.GetIsAromatic()),
            safe_idx(VOCAB["num_rings"], ring_info.NumAtomRings(atom.GetIdx())),
        ],
        dtype=torch.long,
    )

def bond_feature_vec(bond: Chem.Bond) -> torch.Tensor:
    """
    Generate a one-hot feature vector for an RDKit bond type.

    Args:
        bond (Chem.Bond): The input bond.

    Returns:
        torch.Tensor: One-hot encoded feature vector.
    """
    btype = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
    return F.one_hot(torch.tensor(btype), num_classes=4).float()

def fetch_heavy_atom_names_cif(ccd_id: str, timeout: float) -> list:
    """
    Fetch heavy atom names for a specific Chemical Component Dictionary ID.

    Args:
        ccd_id (str): The chemical component ID.
        timeout (float): Request timeout in seconds.

    Returns:
        list of str: Ordered list of heavy atom names.
    """
    url = RCSB_CIF_URL.format(ccd=ccd_id.upper())
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    lines = r.text.splitlines()
    names = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if not any(h.startswith("_chem_comp_atom.") for h in headers):
            continue
        atom_col = next((k for k, h in enumerate(headers) if h == "_chem_comp_atom.atom_id"), None)
        type_col = next((k for k, h in enumerate(headers) if h == "_chem_comp_atom.type_symbol"), None)
        if atom_col is None:
            continue
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("_") or line.startswith("loop_") or line.startswith("#"):
                break
            cols = line.split()
            if len(cols) > atom_col:
                atom_name = cols[atom_col].strip("\"'")
                if type_col is not None and len(cols) > type_col and cols[type_col].upper() == "H":
                    i += 1
                    continue
                names.append(atom_name)
            i += 1
    return names

def fetch_ideal_mol(ccd_id: str, timeout: float) -> Chem.Mol:
    """
    Fetch the ideal RDKit molecule for a given Chemical Component Dictionary ID.

    Args:
        ccd_id (str): The chemical component ID.
        timeout (float): Request timeout in seconds.

    Returns:
        Chem.Mol: The parsed RDKit molecule object.
    """
    url = RCSB_SDF_URL.format(ccd=ccd_id.upper())
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    suppl = Chem.SDMolSupplier()
    suppl.SetData(r.text, removeHs=False)
    mol_h = next(iter(suppl), None)
    if mol_h is None:
        raise ValueError(f"Failed to parse SDF for {ccd_id}")
    mol = Chem.RemoveHs(mol_h, sanitize=True)
    Chem.SanitizeMol(mol)
    return mol
