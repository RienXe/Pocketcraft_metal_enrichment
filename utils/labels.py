"""Module for computing labels and target features for metal binding sites."""

import torch
from utils.config import VOCAB, METAL_ATOMIC_NUMS
from utils.chemistry import fetch_heavy_atom_names_cif, fetch_ideal_mol, atom_feature_vec, bond_feature_vec
from utils.chemistry import safe_idx

def compute_ligand_specific_labels(
    atom_coords_tensor,
    surface_coords,
    ligand_atoms_subset,
    threshold,
    per_residue_atom_indices=None,
    heavy_idx_tensor=None,
):
    """
    Compute binary labels indicating proximity of protein atoms or surface points to a ligand.

    Args:
        atom_coords_tensor (torch.Tensor): Coordinates of all protein atoms.
        surface_coords (torch.Tensor): Coordinates of protein surface points.
        ligand_atoms_subset (AtomArray): Biotite atom array of the ligand.
        threshold (float): Distance threshold for a positive label.
        per_residue_atom_indices (list of list, optional): Mapping of residues to atom indices. Defaults to None.
        heavy_idx_tensor (torch.Tensor, optional): Indices of heavy atoms. Defaults to None.

    Returns:
        dict or None: A dictionary containing atom, surface, and optionally residue labels.
        Returns None if the ligand subset is empty.
    """
    if ligand_atoms_subset is None or len(ligand_atoms_subset) == 0:
        return None

    ligand_coords = torch.tensor(ligand_atoms_subset.coord).float()

    dist_atom  = torch.cdist(atom_coords_tensor, ligand_coords)
    atom_close = (dist_atom.min(dim=1)[0] <= threshold)
    atom_y     = atom_close.float()

    if len(surface_coords) > 0:
        dist_surf = torch.cdist(surface_coords, ligand_coords)
        y = (dist_surf.min(dim=1)[0] <= threshold).float()
    else:
        y = torch.empty(0)

    if heavy_idx_tensor is not None:
        atom_y_heavy = atom_y[heavy_idx_tensor]
    else:
        atom_y_heavy = atom_y

    out = {
        "y":            y,
        "atom_y":       atom_y,
        "atom_y_heavy": atom_y_heavy,
    }

    if per_residue_atom_indices is not None:
        R = len(per_residue_atom_indices)
        residue_y = torch.zeros(R, dtype=torch.float32)
        for r_idx, atom_idxs in enumerate(per_residue_atom_indices):
            if len(atom_idxs) > 0 and atom_close[atom_idxs].any():
                residue_y[r_idx] = 1.0
        out["residue_y"] = residue_y

    return out

def build_structure_coord_map(ligand_atoms_for_residue):
    """
    Create a mapping from atom names to coordinates for a given ligand structure.

    Args:
        ligand_atoms_for_residue (AtomArray): Ligand atoms.

    Returns:
        dict: Dictionary mapping string atom names to 3D coordinate tensors.
    """
    coord_map = {}
    for a in ligand_atoms_for_residue:
        atom_name = str(a.atom_name).strip()
        if atom_name not in coord_map:
            coord_map[atom_name] = torch.tensor(a.coord, dtype=torch.float32)
        if not atom_name.endswith("'") and (atom_name + "'") not in coord_map:
            coord_map[atom_name + "'"] = torch.tensor(a.coord, dtype=torch.float32)
    return coord_map

def extract_single_atom_metal_feature(ccd_id, label_dict_detail, coord_map):
    """
    Extract graph features for a single-atom metal ion.

    Args:
        ccd_id (str): Chemical component ID of the metal.
        label_dict_detail (dict): Mapping of atom names to part IDs.
        coord_map (dict): Mapping of atom names to 3D coordinates.

    Returns:
        dict or None: Graph features for the metal, or None if the atomic number is unknown.
    """
    atomic_num = METAL_ATOMIC_NUMS.get(ccd_id)
    if atomic_num is None:
        return None
    atom_names = list(label_dict_detail.keys())
    if len(atom_names) == 0:
        atom_names = [ccd_id]
    atom_names = [n for n in atom_names if n in coord_map] or atom_names[:1]

    parts_out = {}
    for part_id in sorted(set(label_dict_detail.values())):
        part_names = [n for n, p in label_dict_detail.items() if p == part_id]
        if not part_names:
            continue
        selected = [n for n in part_names if n in coord_map]
        if len(selected) == 0:
            selected = [part_names[0]]

        x_rows = []
        pos_rows = []
        for n in selected:
            x_rows.append([
                safe_idx(VOCAB["atomic_num"], atomic_num),
                safe_idx(VOCAB["chirality"], "CHI_UNSPECIFIED"),
                safe_idx(VOCAB["degree"], 0),
                safe_idx(VOCAB["formal_charge"], 0),
                safe_idx(VOCAB["num_Hs"], 0),
                safe_idx(VOCAB["num_radical_e"], 0),
                safe_idx(VOCAB["hybridization"], "SP3"),
                VOCAB["is_aromatic"].index(False),
                safe_idx(VOCAB["num_rings"], 0),
            ])
            if n in coord_map:
                pos_rows.append(coord_map[n])
            else:
                pos_rows.append(torch.tensor([float("nan")] * 3, dtype=torch.float32))

        parts_out[part_id] = {
            "atom_names": selected,
            "x": torch.tensor(x_rows, dtype=torch.long),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0, 4), dtype=torch.float32),
            "atom_pos_structure": torch.stack(pos_rows),
        }
    return parts_out if parts_out else None

def extract_ligand_graph_features(ccd_id, ligand_atoms_for_residue, label_dict_detail, timeout):
    """
    Extract RDKit-based graph features for a complex ligand.

    Fetches the ideal SDF and CIF from RCSB to build a molecular graph with bonds
    and atomic properties, mapped to the provided 3D coordinates.

    Args:
        ccd_id (str): Chemical component ID of the ligand.
        ligand_atoms_for_residue (AtomArray): Ligand structure parsed from PDB/CIF.
        label_dict_detail (dict): Mapping of atom names to part IDs.
        timeout (float): Request timeout for RCSB.

    Returns:
        dict: A dictionary mapping part IDs to their respective graph features.
    """
    coord_map = build_structure_coord_map(ligand_atoms_for_residue)
    try:
        ordered_names = fetch_heavy_atom_names_cif(ccd_id, timeout=timeout)
        mol = fetch_ideal_mol(ccd_id, timeout=timeout)
        if mol.GetNumAtoms() != len(ordered_names):
            raise ValueError(f"Atom count mismatch ({mol.GetNumAtoms()} vs {len(ordered_names)})")

        rdkit_to_name = {i: ordered_names[i] for i in range(mol.GetNumAtoms())}
        name_to_rdkit = {v: k for k, v in rdkit_to_name.items()}
        for name, idx in list(name_to_rdkit.items()):
            if name + "'" not in name_to_rdkit:
                name_to_rdkit[name + "'"] = idx

        ring_info = mol.GetRingInfo()
        parts_out = {}

        for part_id in sorted(set(label_dict_detail.values())):
            part_names = [n for n, p in label_dict_detail.items() if p == part_id and n in name_to_rdkit]
            if len(part_names) == 0:
                continue
            local_idx = {n: i for i, n in enumerate(part_names)}
            x = torch.stack([atom_feature_vec(mol.GetAtomWithIdx(name_to_rdkit[n]), ring_info) for n in part_names])

            structure_pos = []
            for n in part_names:
                c = coord_map.get(n)
                if c is None:
                    c = coord_map.get(n[:-1]) if n.endswith("'") else coord_map.get(n + "'")
                if c is None:
                    c = torch.tensor([float("nan")] * 3, dtype=torch.float32)
                structure_pos.append(c)

            rows, cols, edge_attrs = [], [], []
            for bond in mol.GetBonds():
                i_name = rdkit_to_name[bond.GetBeginAtomIdx()]
                j_name = rdkit_to_name[bond.GetEndAtomIdx()]
                i_local = local_idx.get(i_name) if i_name in local_idx else local_idx.get(i_name + "'")
                j_local = local_idx.get(j_name) if j_name in local_idx else local_idx.get(j_name + "'")
                if i_local is not None and j_local is not None:
                    bf = bond_feature_vec(bond)
                    rows += [i_local, j_local]
                    cols += [j_local, i_local]
                    edge_attrs += [bf, bf]

            edge_index = (
                torch.tensor([rows, cols], dtype=torch.long) if rows else torch.empty((2, 0), dtype=torch.long)
            )
            edge_attr = (
                torch.stack(edge_attrs) if edge_attrs else torch.empty((0, 4), dtype=torch.float32)
            )
            parts_out[part_id] = {
                "atom_names": part_names,
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "atom_pos_structure": torch.stack(structure_pos),
            }

        if len(parts_out) > 0:
            return parts_out
    except Exception:
        pass

    return extract_single_atom_metal_feature(ccd_id, label_dict_detail, coord_map)
