"""Module for extracting and calculating geometric and chemical features."""

import numpy as np
import torch
import torch.nn.functional as F
import biotite.structure as struc
import biotite.structure.info as info

def get_raw_atom_feature(res_name, atom_name, element, aa_feat_dict, atom_feat_dict):
    """
    Retrieve raw categorical feature value for an atom, prioritizing amino acid specific rules.

    Args:
        res_name (str): Three-letter residue name.
        atom_name (str): Atom name (e.g., 'CA', 'CB').
        element (str): Element symbol.
        aa_feat_dict (dict): Dictionary of amino acid specific features.
        atom_feat_dict (dict): Dictionary of generic atom features by element.

    Returns:
        int or float: The feature value.
    """
    if res_name in aa_feat_dict and atom_name in aa_feat_dict[res_name]:
        return aa_feat_dict[res_name][atom_name]
    return atom_feat_dict.get(element, 0)

def fibonacci_sphere(samples=1):
    """
    Generate points on a unit sphere using a Fibonacci spiral.

    Args:
        samples (int, optional): Number of points to generate. Defaults to 1.

    Returns:
        numpy.ndarray: Array of 3D coordinates of shape (samples, 3).
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def generate_surface_points(protein_atoms, num_points_per_atom=20, probe_radius=1.4):
    """
    Generate solvent accessible surface points around a protein.

    This function first pre-filters atoms based on solvent accessible surface area (SASA),
    then generates points around them, and removes points that lie inside the van der Waals
    radius (plus probe radius) of any neighboring atom.

    Args:
        protein_atoms (AtomArray): Biotite atom array of the protein.
        num_points_per_atom (int, optional): Number of initial points to generate per surface atom. Defaults to 20.
        probe_radius (float, optional): Radius of the solvent probe. Defaults to 1.4.

    Returns:
        torch.Tensor: Filtered surface point coordinates of shape [N, 3].
    """
    atom_sasa = struc.sasa(protein_atoms, probe_radius=probe_radius)
    surface_atom_mask = atom_sasa > 0
    surface_atoms = protein_atoms[surface_atom_mask]

    if len(surface_atoms) == 0:
        return torch.empty((0, 3))

    radii = np.array([info.vdw_radius_single(e) + probe_radius for e in surface_atoms.element])
    atom_coords = surface_atoms.coord
    unit_sphere_pts = fibonacci_sphere(num_points_per_atom)

    all_pts = atom_coords[:, np.newaxis, :] + unit_sphere_pts[np.newaxis, :, :] * radii[:, np.newaxis, np.newaxis]
    all_pts = all_pts.reshape(-1, 3)

    all_radii = np.array([info.vdw_radius_single(e) + probe_radius for e in protein_atoms.element])
    all_coords = protein_atoms.coord
    cell_list = struc.CellList(protein_atoms, cell_size=np.max(all_radii))

    keep_mask = np.ones(len(all_pts), dtype=bool)
    for i, p in enumerate(all_pts):
        neighbor_indices = cell_list.get_atoms(p, radius=np.max(all_radii))
        dists_sq = np.sum((p - all_coords[neighbor_indices]) ** 2, axis=1)
        if np.any(dists_sq < (all_radii[neighbor_indices] - 0.01) ** 2):
            keep_mask[i] = False

    return torch.tensor(all_pts[keep_mask]).float()

def build_per_atom_feature_tensor(protein_atoms, feat_config):
    """
    Build a feature tensor for all atoms in a protein.

    Combines chemical properties and amino acid identities into a one-hot encoded tensor.

    Args:
        protein_atoms (AtomArray): Biotite atom array of the protein.
        feat_config (dict): Feature configuration dictionaries.

    Returns:
        torch.Tensor: Atom feature tensor of shape [num_atoms, 32].
    """
    aa_feat_dict   = feat_config["dict_aa_feature"]
    atom_feat_dict = feat_config["dict_atom_feature"]
    regular_aa     = feat_config["regular_aa"]
    n_aa           = len(regular_aa)
    unk_idx        = regular_aa.index("UNK")

    val_idx_list = []
    aa_idx_list  = []
    for a in protein_atoms:
        v = get_raw_atom_feature(a.res_name, a.atom_name, a.element,
                                 aa_feat_dict, atom_feat_dict)
        val_idx_list.append(int(v))
        aa_idx_list.append(
            regular_aa.index(a.res_name) if a.res_name in regular_aa else unk_idx
        )

    val_idx = torch.tensor(val_idx_list, dtype=torch.long)
    aa_idx  = torch.tensor(aa_idx_list,  dtype=torch.long)
    val_oh = F.one_hot(val_idx, num_classes=11).float()
    aa_oh  = F.one_hot(aa_idx,  num_classes=n_aa).float()
    return torch.cat([val_oh, aa_oh], dim=-1)  # [N_atoms, 32]

def get_weighted_features(surface_pts, atom_pts, atom_feats, k=3):
    """
    Interpolate atom features onto surface points based on inverse distance weighting.

    Args:
        surface_pts (torch.Tensor): Coordinates of surface points.
        atom_pts (torch.Tensor): Coordinates of protein atoms.
        atom_feats (torch.Tensor): Features of protein atoms.
        k (int, optional): Number of nearest neighbors to use. Defaults to 3.

    Returns:
        torch.Tensor: Interpolated features at each surface point.
    """
    dist = torch.cdist(surface_pts, atom_pts)
    top_dist, indices = torch.topk(dist, k, largest=False)
    weights = 1.0 / (top_dist + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (atom_feats[indices] * weights.unsqueeze(-1)).sum(dim=1)

def extract_residue_level_features(protein_atoms):
    """
    Extract residue-level features and metadata from an atom array.

    Args:
        protein_atoms (AtomArray): Biotite atom array of the protein.

    Returns:
        dict: A dictionary containing residue names, indices, C-beta coordinates,
        distance matrices, and per-residue atom index mappings.
    """
    residue_names = []
    residue_indices = []
    residue_chain_ids = []
    cb_coords = []
    cb_valid = []
    per_residue_atom_indices = []

    atom_offset = 0
    for residue in struc.residue_iter(protein_atoms):
        n_atoms_in_res = len(residue)

        res_name = str(residue.res_name[0])
        res_id   = int(residue.res_id[0])
        chain_id = str(residue.chain_id[0])

        residue_names.append(res_name)
        residue_indices.append(res_id)
        residue_chain_ids.append(chain_id)

        atom_names = [str(n).strip() for n in residue.atom_name]

        cb_idx = None
        is_real_cb = False
        if "CB" in atom_names:
            cb_idx = atom_names.index("CB")
            is_real_cb = True
        elif "CA" in atom_names:
            cb_idx = atom_names.index("CA")
            is_real_cb = False

        if cb_idx is not None:
            cb_coords.append(torch.tensor(residue.coord[cb_idx], dtype=torch.float32))
            cb_valid.append(is_real_cb)
        else:
            cb_coords.append(torch.tensor([float("nan")] * 3, dtype=torch.float32))
            cb_valid.append(False)

        per_residue_atom_indices.append(
            list(range(atom_offset, atom_offset + n_atoms_in_res))
        )
        atom_offset += n_atoms_in_res

    R = len(residue_names)

    cb_coords_tensor = torch.stack(cb_coords) if cb_coords else torch.empty((0, 3))
    cb_valid_tensor  = torch.tensor(cb_valid, dtype=torch.bool)
    res_idx_tensor   = torch.tensor(residue_indices, dtype=torch.long)

    if R > 0:
        cb_dist = torch.cdist(cb_coords_tensor, cb_coords_tensor)
    else:
        cb_dist = torch.empty((0, 0))

    return {
        "residue_names":              residue_names,
        "residue_indices":            res_idx_tensor,
        "residue_chain_ids":          residue_chain_ids,
        "cb_coords":                  cb_coords_tensor,
        "cb_valid_mask":              cb_valid_tensor,
        "cb_dist_matrix":             cb_dist,
        "_per_residue_atom_indices":  per_residue_atom_indices,
    }
