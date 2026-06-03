"""Module for evaluating spatial predictions of metal binding sites against ground truth structures."""

import numpy as np
from scipy.spatial import distance
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

def load_structure(file_path):
    """
    Load a protein structure from a CIF or PDB file, extracting B-factors if available.

    Args:
        file_path (str): Path to the input structure file.

    Returns:
        AtomArray: Parsed biotite atom array.
    """
    if file_path.endswith(".cif"):
        cif = pdbx.CIFFile.read(file_path)
        try:
            atoms = pdbx.get_structure(cif, model=1, extra_fields=["b_factor"])
        except Exception:
            atoms = pdbx.get_structure(cif, model=1, data_block="PREDICTED", extra_fields=["b_factor"])
    else:
        pdb_f = pdb.PDBFile.read(file_path)
        atoms = pdb.get_structure(pdb_f, model=1, extra_fields=["b_factor"])
    return atoms

def extract_protein_and_metals(atoms, target_metals):
    """
    Separate a structure into protein atoms and specific metal target atoms.

    Args:
        atoms (AtomArray): Biotite atom array containing the full structure.
        target_metals (list of str): List of elemental symbols to extract as metals.

    Returns:
        tuple: A tuple containing two AtomArrays:
            - prot_atoms (AtomArray): Protein atoms.
            - metal_atoms (AtomArray): Target metal atoms.
    """
    if len(atoms) == 0: return atoms, atoms
    prot_mask = struc.filter_amino_acids(atoms) & (atoms.element != "H")
    metal_mask = atoms.hetero & np.isin(atoms.element, target_metals)
    return atoms[prot_mask], atoms[metal_mask]

def evaluate_task(gt_coords, gt_cats, gt_types, pred_coords, pred_types, pred_probs, pdb_id, prot_atoms, task="site"):
    """
    Evaluate predicted metal binding sites against ground truth points.

    Args:
        gt_coords (numpy.ndarray): Coordinates of ground truth metals.
        gt_cats (list of str): Categories for each ground truth site (e.g., 'functional').
        gt_types (list of str): Elemental types of ground truth metals.
        pred_coords (numpy.ndarray): Coordinates of predicted metals.
        pred_types (numpy.ndarray): Elemental types of predicted metals.
        pred_probs (numpy.ndarray): Confidence probabilities for each prediction.
        pdb_id (str): Identifier of the current structure.
        prot_atoms (AtomArray): Original protein structure.
        task (str, optional): Evaluation strategy ('site', 'exact', or 'top1'). Defaults to "site".

    Returns:
        list of dict: Detailed evaluation results for each ground truth and prediction point.
    """
    results = []
    matched_preds = {}
    prot_coords = prot_atoms.coord if len(prot_atoms) > 0 else np.array([])

    for i, gt_c in enumerate(gt_coords):
        cat = gt_cats[i]
        gt_t = gt_types[i]
        is_pos = 1 if cat in ['functional', 'possible'] else 0
        
        if len(pred_coords) > 0:
            dists = distance.cdist([gt_c], pred_coords)[0]
            close_idx = np.where(dists <= 2.0)[0]

            if len(close_idx) > 0:
                if task == "exact":
                    close_types = pred_types[close_idx]
                    
                    if is_pos and (gt_t in close_types):
                        status = "TP"
                        correct_type_indices = close_idx[close_types == gt_t]
                        top_idx = correct_type_indices[np.argmax(pred_probs[correct_type_indices])]
                        top_type = gt_t
                    else:
                        top_idx = close_idx[np.argmax(pred_probs[close_idx])]
                        top_type = pred_types[top_idx]
                        status = "FN_WrongType" if is_pos else "FP_Negative"
                else:
                    top_idx = close_idx[np.argmax(pred_probs[close_idx])]
                    top_type = pred_types[top_idx]
                    
                    if task == "site":
                        status = "TP" if is_pos else "FP_Negative"
                    elif task == "top1":
                        status = "TP" if (is_pos and top_type == gt_t) else ("FN_WrongType" if is_pos else "FP_Negative")

                results.append({
                    'pdb_id': pdb_id, 'cat': cat, 'y_true': is_pos, 
                    'y_score': pred_probs[top_idx] if status == "TP" else 0.0, 
                    'dist': dists[top_idx], 'status': status, 
                    'gt_type': gt_t, 'pred_type': top_type
                })
                matched_preds[top_idx] = is_pos
            else:
                status = "FN_Missed" if is_pos else "TN"
                results.append({'pdb_id': pdb_id, 'cat': cat, 'y_true': is_pos, 'y_score': 0.0, 'dist': None, 'status': status, 'gt_type': gt_t, 'pred_type': None})
        else:
            status = "FN_Missed" if is_pos else "TN"
            results.append({'pdb_id': pdb_id, 'cat': cat, 'y_true': is_pos, 'y_score': 0.0, 'dist': None, 'status': status, 'gt_type': gt_t, 'pred_type': None})
            
    for j in range(len(pred_coords)):
        if j not in matched_preds:
            p_coord = pred_coords[j]
            pred_cat = 'Negative' 
            
            if len(prot_coords) > 0:
                dist_to_prot = distance.cdist([p_coord], prot_coords)[0]
                if np.any(dist_to_prot <= 3.0):
                    close_prot_mask = dist_to_prot <= 2.2
                    if np.sum(close_prot_mask) > 0:
                        close_atoms = prot_atoms[close_prot_mask]
                        unique_residues = set(zip(close_atoms.chain_id, close_atoms.res_id))
                        pred_cat = 'functional' if len(unique_residues) >= 3 else 'possible'
                    else:
                        pred_cat = 'possible'
                else:
                    pred_cat = 'Negative'

            results.append({
                'pdb_id': pdb_id, 
                'cat': pred_cat,
                'y_true': 0, 
                'y_score': pred_probs[j], 
                'dist': None, 
                'status': 'FP_Background', 
                'gt_type': None, 
                'pred_type': pred_types[j]
            })
            
    return results

def process_single_pair(gt_path, pred_path, target_metals):
    """
    Process a single ground-truth and prediction file pair to calculate metrics.

    Args:
        gt_path (str): Path to the ground truth structure.
        pred_path (str): Path to the predicted structure.
        target_metals (list of str): List of metal elements to evaluate.

    Returns:
        tuple: A tuple containing lists of results for different tasks and a summary count dict:
            - site_res (list): Results for site-level detection.
            - exact_res (list): Results for exact type matching.
            - top1_res (list): Results for top-1 probability predictions.
            - counts (dict): Summary statistics of ground truth and predicted metals.
    """
    import os
    base_name = os.path.splitext(os.path.basename(gt_path))[0]
    
    gt_atoms = load_structure(gt_path)
    gt_protein, gt_metals = extract_protein_and_metals(gt_atoms, target_metals)
    
    if os.path.exists(pred_path):
        pred_atoms = load_structure(pred_path)
        _, pred_metals = extract_protein_and_metals(pred_atoms, target_metals)
    else:
        pred_metals = struc.AtomArray(0)
        
    counts = {
        'gt_all': len(gt_metals),
        'gt_valid': 0,
        'gt_functional': 0,
        'gt_possible': 0,
        'gt_negative': 0,
        'pred_all': len(pred_metals)
    }
        
    if len(gt_metals) == 0:
        if len(pred_metals) > 0:
            pred_coords = pred_metals.coord
            pred_probs = pred_metals.b_factor / 100.0 if hasattr(pred_metals, "b_factor") else np.zeros(len(pred_metals))
            pred_types = pred_metals.element
            site_res = evaluate_task([], [], [], pred_coords, pred_types, pred_probs, base_name, gt_protein, task="site")
            exact_res = evaluate_task([], [], [], pred_coords, pred_types, pred_probs, base_name, gt_protein, task="exact")
            top1_res = evaluate_task([], [], [], pred_coords, pred_types, pred_probs, base_name, gt_protein, task="top1")
            return site_res, exact_res, top1_res, counts
        return None, None, None, counts

    gt_coords = gt_metals.coord
    prot_coords = gt_protein.coord
    pred_coords = pred_metals.coord if len(pred_metals) > 0 else np.array([])
    
    if len(pred_metals) > 0 and hasattr(pred_metals, "b_factor"):
        pred_probs = pred_metals.b_factor / 100.0
    else:
        pred_probs = np.zeros(len(pred_metals))
        
    pred_types = pred_metals.element if len(pred_metals) > 0 else np.array([])
    
    gt_cats = []
    for i, gt_coord in enumerate(gt_coords):
        if len(prot_coords) == 0: 
            gt_cats.append('Negative')
            counts['gt_negative'] += 1
            continue
            
        dist_to_prot = distance.cdist([gt_coord], prot_coords)[0]
        if np.sum(dist_to_prot <= 3.0) == 0:
            gt_cats.append('Negative')
            counts['gt_negative'] += 1
        else:
            counts['gt_valid'] += 1
            close_prot_mask = dist_to_prot <= 2.2
            if np.sum(close_prot_mask) > 0:
                close_atoms = gt_protein[close_prot_mask]
                unique_residues = set(zip(close_atoms.chain_id, close_atoms.res_id))
                if len(unique_residues) >= 3:
                    gt_cats.append('functional'); counts['gt_functional'] += 1
                else:
                    gt_cats.append('possible'); counts['gt_possible'] += 1
            else:
                gt_cats.append('possible'); counts['gt_possible'] += 1

    gt_types = gt_metals.element
    gt_cats = np.array(gt_cats)

    site_res = evaluate_task(gt_coords, gt_cats, gt_types, pred_coords, pred_types, pred_probs, base_name, gt_protein, task="site")
    exact_res = evaluate_task(gt_coords, gt_cats, gt_types, pred_coords, pred_types, pred_probs, base_name, gt_protein, task="exact")
    top1_res = evaluate_task(gt_coords, gt_cats, gt_types, pred_coords, pred_types, pred_probs, base_name, gt_protein, task="top1")

    return site_res, exact_res, top1_res, counts
