"""Module for preprocessing PDB/CIF structures into PyTorch Geometric datasets."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch
from tqdm import tqdm

from utils.config import load_config, FEATURE_CONFIG_PATH, DEFAULT_LIGAND_FEATURE_JSON
from utils.features import build_per_atom_feature_tensor, generate_surface_points, get_weighted_features, extract_residue_level_features
from utils.labels import compute_ligand_specific_labels, extract_ligand_graph_features

def process_file(
    filename, input_dir, output_dir, feat_config, points_per_atom, threshold,
    ligand_dir, probe_radius, ligand_feature_defs, include_ligand_graph,
    http_timeout, include_residue_features, include_atom_features, drop_hydrogens, max_residues
):
    """
    Process a single structure file to extract features and labels.

    Args:
        filename (str): Name of the file to process.
        input_dir (str): Directory containing input structures.
        output_dir (str): Directory to save processed PyTorch files.
        feat_config (dict): Feature configuration.
        points_per_atom (int): Points to generate per atom for surface processing.
        threshold (float): Distance threshold for positive labels.
        ligand_dir (str or None): Directory containing isolated ligands.
        probe_radius (float): Radius for surface generation.
        ligand_feature_defs (dict or None): Ligand graph feature definitions.
        include_ligand_graph (bool): Whether to include graph features.
        http_timeout (float): Timeout for RCSB requests.
        include_residue_features (bool): Whether to extract residue-level features.
        include_atom_features (bool): Whether to keep atom-level features.
        drop_hydrogens (bool): Whether to drop hydrogen atoms.
        max_residues (int or None): Skip structures exceeding this limit.

    Returns:
        bool or str or None: True if successful, None if skipped, or an error string.
    """
    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, base_name + ".pt")

    if os.path.exists(output_path):
        return None

    try:
        if filename.endswith(".cif"):
            cif_file = pdbx.CIFFile.read(input_path)
            atoms = pdbx.get_structure(cif_file, model=1)
        elif filename.endswith(".pdb"):
            pdb_file = pdb.PDBFile.read(input_path)
            atoms = pdb.get_structure(pdb_file, model=1)
        else:
            return f"Unsupported file format: {filename}"

        protein_atoms = atoms[struc.filter_amino_acids(atoms)]

        if max_residues is not None and max_residues > 0:
            n_residues = struc.get_residue_count(protein_atoms)
            if n_residues > max_residues:
                return f"{filename}: skipped ({n_residues} residues > max_residues={max_residues})"

        atom_features_full = build_per_atom_feature_tensor(protein_atoms, feat_config)

        surface_coords = generate_surface_points(protein_atoms, points_per_atom, probe_radius)
        atom_coords_tensor = torch.tensor(protein_atoms.coord).float()

        if len(surface_coords) > 0:
            cloud_features = get_weighted_features(surface_coords, atom_coords_tensor, atom_features_full)
        else:
            cloud_features = torch.empty((0, 11 + len(feat_config["regular_aa"])))

        ligand_atoms = None
        if ligand_dir:
            ligand_path = os.path.join(ligand_dir, filename)
            if os.path.exists(ligand_path):
                if ligand_path.endswith(".cif"):
                    ligand_cif = pdbx.CIFFile.read(ligand_path)
                    ligand_atoms_all = pdbx.get_structure(ligand_cif, model=1)
                elif ligand_path.endswith(".pdb"):
                    ligand_pdb = pdb.PDBFile.read(ligand_path)
                    ligand_atoms_all = pdb.get_structure(ligand_pdb, model=1)
                else:
                    ligand_atoms_all = None
                if ligand_atoms_all is not None:
                    ligand_atoms = ligand_atoms_all[~struc.filter_solvent(ligand_atoms_all)]
        else:
            ligand_mask = atoms.hetero & ~struc.filter_solvent(atoms) & ~struc.filter_amino_acids(atoms)
            ligand_atoms = atoms[ligand_mask]

        data = {
            "x":        cloud_features,
            "pos":      surface_coords,
            "atom_pos": atom_coords_tensor,
        }

        heavy_idx_tensor = None
        if include_atom_features:
            if drop_hydrogens:
                heavy_mask = (protein_atoms.element != "H")
                heavy_idx = np.where(heavy_mask)[0]
                heavy_idx_tensor = torch.from_numpy(heavy_idx).long()

                atom_x         = atom_features_full[heavy_idx_tensor]
                atom_pos_heavy = atom_coords_tensor[heavy_idx_tensor]
                heavy_atoms    = protein_atoms[heavy_mask]
            else:
                atom_x         = atom_features_full
                atom_pos_heavy = atom_coords_tensor
                heavy_atoms    = protein_atoms

            data["atom_x"]         = atom_x                                            
            data["atom_pos_heavy"] = atom_pos_heavy                                    
            data["atom_names"]     = np.array([str(a.atom_name) for a in heavy_atoms])
            data["atom_res_names"] = np.array([str(a.res_name)  for a in heavy_atoms])
            data["atom_res_ids"]   = torch.tensor(np.array(heavy_atoms.res_id), dtype=torch.long)
            data["atom_chain_ids"] = np.array([str(a.chain_id) for a in heavy_atoms])
            data["atom_element"]   = np.array([str(a.element)  for a in heavy_atoms])

        per_residue_atom_indices = None
        if include_residue_features:
            residue_data = extract_residue_level_features(protein_atoms=protein_atoms)
            per_residue_atom_indices = residue_data.pop("_per_residue_atom_indices")
            data.update(residue_data)

        ligand_features = {}
        if ligand_atoms is not None and len(ligand_atoms) > 0:
            for lig_sub in struc.residue_iter(ligand_atoms):
                ccd_id = str(lig_sub.res_name[0])
                chain_id = str(lig_sub.chain_id[0])
                res_id = str(lig_sub.res_id[0])
                
                ins_code = ""
                if hasattr(lig_sub, "ins_code"):
                    ins = str(lig_sub.ins_code[0]).strip()
                    if ins:
                        ins_code = f"_{ins}"
                
                lig_instance_id = f"{ccd_id}_{chain_id}_{res_id}{ins_code}"

                labels = compute_ligand_specific_labels(
                    atom_coords_tensor=atom_coords_tensor,
                    surface_coords=surface_coords,
                    ligand_atoms_subset=lig_sub,
                    threshold=threshold,
                    per_residue_atom_indices=per_residue_atom_indices,
                    heavy_idx_tensor=heavy_idx_tensor,
                )
                if labels is None:
                    continue

                entry = dict(labels)
                entry["ccd_id"] = ccd_id
                entry["chain_id"] = chain_id
                entry["res_id"] = res_id

                parts = {}
                if include_ligand_graph:
                    if (
                        ligand_feature_defs is not None
                        and ccd_id in ligand_feature_defs
                    ):
                        label_dict = ligand_feature_defs[ccd_id]
                    else:
                        heavy_names_in_struct = {
                            str(a.atom_name).strip()
                            for a in lig_sub
                            if str(a.element).upper() != "H"
                        }
                        label_dict = {n: 0 for n in heavy_names_in_struct}

                    if len(label_dict) > 0:
                        feat = extract_ligand_graph_features(
                            ccd_id=ccd_id,
                            ligand_atoms_for_residue=lig_sub,
                            label_dict_detail=label_dict,
                            timeout=http_timeout,
                        )
                        if feat is not None:
                            parts = feat
                entry["parts"] = parts

                ligand_features[lig_instance_id] = entry

        data["ligand_features"] = ligand_features

        torch.save(data, output_path)
        return True

    except Exception as e:
        return f"{filename}: {str(e)}"

def main():
    """
    Main entry point for batch preprocessing.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Preprocess protein structures into tensor representations for metal enrichment training.")
    parser.add_argument("--input_dir", required=True, help="Directory containing raw protein structure files (.pdb or .cif).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated PyTorch tensor files (.pt).")
    parser.add_argument("--config", default=FEATURE_CONFIG_PATH, help="Path to the JSON feature configuration file.")
    parser.add_argument("--points_per_atom", type=int, default=20, help="Number of surface points generated per atom.")
    parser.add_argument("--threshold", type=float, default=4.5, help="Distance threshold for positive label assignment.")
    parser.add_argument("--ligand_dir", help="Optional directory containing separate ligand files.")
    parser.add_argument("--probe_radius", type=float, default=1.4, help="Probe radius for Solvent Accessible Surface (SAS) generation.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of concurrent worker processes.")
    parser.add_argument("--include_ligand_graph", action="store_true", help="Include ligand graph features in the output tensors.")
    parser.add_argument("--ligand_feature_json", default=DEFAULT_LIGAND_FEATURE_JSON, help="Path to the JSON ligand definitions file.")
    parser.add_argument("--http_timeout", type=float, default=20.0, help="Timeout in seconds for RCSB HTTP requests.")
    parser.add_argument("--no_residue_features", action="store_true", help="Omit residue-level features from the output.")
    parser.add_argument("--no_atom_features", action="store_true", help="Omit per-atom features from the output.")
    parser.add_argument("--keep_hydrogens", action="store_true", help="Retain hydrogen atoms in the final tensor representation.")
    parser.add_argument("--max_residues", type=int, default=1000, help="Maximum number of residues allowed; larger proteins are skipped.")

    args = parser.parse_args()
    feat_config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    ligand_feature_defs = None
    if args.include_ligand_graph:
        if not os.path.exists(args.ligand_feature_json):
            raise FileNotFoundError(f"ligand_feature_json not found: {args.ligand_feature_json}")
        combined = load_config(args.ligand_feature_json)
        ligand_feature_defs = combined.get("label_dict_detail", {})
        print(f"Loaded ligand feature definitions: {len(ligand_feature_defs)} CCD IDs")

    include_residue = not args.no_residue_features
    include_atom    = not args.no_atom_features
    drop_h          = not args.keep_hydrogens
    max_res         = args.max_residues if args.max_residues > 0 else None

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith((".cif", ".pdb"))]

    process_func = partial(
        process_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        feat_config=feat_config,
        points_per_atom=args.points_per_atom,
        threshold=args.threshold,
        ligand_dir=args.ligand_dir,
        probe_radius=args.probe_radius,
        ligand_feature_defs=ligand_feature_defs,
        include_ligand_graph=args.include_ligand_graph,
        http_timeout=args.http_timeout,
        include_residue_features=include_residue,
        include_atom_features=include_atom,
        drop_hydrogens=drop_h,
        max_residues=max_res,
    )

    print(f"Processing {len(input_files)} files using {args.num_workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(process_func, input_files), total=len(input_files)))

    skipped_exist  = results.count(None)
    errors_all     = [r for r in results if isinstance(r, str)]
    skipped_size   = [e for e in errors_all if "skipped" in e]
    real_errors    = [e for e in errors_all if "skipped" not in e]
    success        = results.count(True)

    print("\nProcessing complete:")
    print(f"  - Successfully processed: {success}")
    print(f"  - Skipped (already exist): {skipped_exist}")
    print(f"  - Skipped (oversized):     {len(skipped_size)}")
    print(f"  - Errors:                  {len(real_errors)}")

if __name__ == "__main__":
    main()
