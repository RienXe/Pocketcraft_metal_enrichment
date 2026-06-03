"""Module for batch predicting metal binding sites using a trained model."""

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import HeteroData

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

from utils.config import METAL_VOCABULARY, METAL_TO_IDX, load_config, FEATURE_CONFIG_PATH
from models.predictor import MetalBindingPredictor
from utils.features import build_per_atom_feature_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_safe(checkpoint_path, hidden_dim):
    """
    Safely load a trained model checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        hidden_dim (int): Default hidden dimension if not found in checkpoint args.

    Returns:
        MetalBindingPredictor: The loaded and initialized model.
    """
    torch.serialization.add_safe_globals([
        np.dtype, 
        np._core.multiarray.scalar, 
        np.ndarray, 
        np._core.multiarray._reconstruct
    ])
    
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    hyperparams = ckpt.get("args", {})
    if "args" in ckpt:
        hidden_dim = hyperparams.get("hidden_dim", hidden_dim)
        num_rbf = hyperparams.get("num_rbf", 16)
        cutoff = hyperparams.get("cutoff", 10.0)
        num_layers = hyperparams.get("num_layers", 4)
        max_neighbors = hyperparams.get("max_neighbors", 32)
        dropout = hyperparams.get("dropout", 0.1)
    else:
        num_rbf = 16
        cutoff = 10.0
        num_layers = 4
        max_neighbors = 32
        dropout = 0.1
    
    model = MetalBindingPredictor(
        protein_in=32, 
        hidden_dim=hidden_dim,
        num_rbf=num_rbf,
        cutoff=cutoff,
        num_layers=num_layers,
        max_neighbors=max_neighbors,
        dropout=dropout
    ).to(DEVICE)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def prepare_protein_for_prediction(file_path, feat_config):
    """
    Prepare protein features from a structure file.

    Args:
        file_path (str): Path to the input CIF or PDB file.
        feat_config (dict): Feature configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - tensors (dict or None): Dictionary of feature tensors.
            - protein_atoms (AtomArray or None): Parsed biotite atom array.
    """
    if file_path.endswith(".cif"):
        cif = pdbx.CIFFile.read(file_path)
        atoms = pdbx.get_structure(cif, model=1)
    else:
        pdb_f = pdb.PDBFile.read(file_path)
        atoms = pdb.get_structure(pdb_f, model=1)

    prot_mask = struc.filter_amino_acids(atoms) & (atoms.element != "H")
    protein_atoms = atoms[prot_mask]
    
    if len(protein_atoms) == 0:
        return None, None

    atom_x = build_per_atom_feature_tensor(protein_atoms, feat_config)

    tensors = {
        "atom_x": atom_x,
        "atom_pos_heavy": torch.tensor(protein_atoms.coord).float(),
    }
    return tensors, protein_atoms

@torch.no_grad()
def predict_structure(model, data, threshold):
    """
    Predict metal binding locations and probabilities for a single structure.

    Args:
        model (MetalBindingPredictor): The trained prediction model.
        data (HeteroData): Geometric data object for the protein.
        threshold (float): Probability threshold for keeping a prediction.

    Returns:
        list of dict: List of prediction dictionaries.
    """
    preds = []
    for m_name in ['ZN', 'CA', 'MG', 'FE', 'MN', 'CU', 'NI', 'CO', 'NA', 'K']:
        data.metal_type = torch.tensor([METAL_TO_IDX[m_name]], device=DEVICE)
        logits, offsets, updated_pos, _ = model(data)
        
        probs = torch.sigmoid(logits).squeeze() 
        final_coords = updated_pos + offsets
        
        mask = probs > threshold
        if mask.any():
            idxs = torch.where(mask)[0]
            for i in idxs:
                preds.append({
                    "metal": m_name, 
                    "coord": final_coords[i].cpu().numpy(),
                    "prob": probs[i].item(), 
                    "logit": logits[i].item()
                })
    return preds

def filter_overlapping_predictions(preds, distance_cutoff=2.0):
    """
    Filter predictions to remove spatially overlapping points of the same metal.

    Args:
        preds (list of dict): Raw predictions.
        distance_cutoff (float, optional): Minimum allowed distance between predictions. Defaults to 2.0.

    Returns:
        list of dict: Filtered list of predictions.
    """
    if not preds:
        return []
    
    preds = sorted(preds, key=lambda x: x['prob'], reverse=True)
    filtered = []
    
    for p in preds:
        is_duplicate = False
        for f in filtered:
            if p['metal'] == f['metal']:
                dist = np.linalg.norm(p['coord'] - f['coord'])
                if dist < distance_cutoff:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append(p)
            
    return filtered

def write_results(original_prot, predictions, out_path):
    """
    Write the original protein and predicted metal atoms to a CIF file.

    Args:
        original_prot (AtomArray): Original protein structure.
        predictions (list of dict): List of valid metal predictions.
        out_path (str): Path to save the combined structure.

    Returns:
        None
    """
    from biotite.structure.io.pdbx import CIFCategory
    
    n_prot = original_prot.array_length()
    n_met = len(predictions)
    
    combined = original_prot + struc.AtomArray(n_met)
    
    existing_chains = set(struc.get_chains(original_prot))
    
    if "M" not in existing_chains:
        target_chain = "M"
    elif "ME" not in existing_chains:
        target_chain = "ME"
    else:
        target_chain = "MET"
    
    b_factors = np.zeros(n_prot + n_met)
    hetero_flags = np.zeros(n_prot + n_met, dtype=bool)
    
    if hasattr(original_prot, "b_factor"): 
        b_factors[:n_prot] = original_prot.b_factor
    hetero_flags[:n_prot] = original_prot.hetero

    for i, p in enumerate(predictions):
        idx = n_prot + i
        combined.coord[idx] = p["coord"]
        combined.atom_name[idx] = f"{p['metal']}{i+1}"
        combined.res_name[idx] = p["metal"]
        combined.res_id[idx] = i + 1 
        combined.chain_id[idx] = target_chain 
        combined.element[idx] = p["metal"][:2].strip()
        
        hetero_flags[idx] = True
        b_factors[idx] = p["prob"] * 100 

    combined.set_annotation("b_factor", b_factors)
    combined.set_annotation("hetero", hetero_flags)
    
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, combined, data_block="PREDICTED")
    
    if n_met > 0:
        stats_dict = {
            "metal": [p["metal"] for p in predictions],
            "confidence": [f"{p['prob']:.4f}" for p in predictions],
            "logit": [f"{p['logit']:.4f}" for p in predictions],
            "residue_id": [str(i + 1) for i in range(n_met)]
        }
        cif["PREDICTED"]["_predicted_metal_scores"] = CIFCategory(stats_dict)
        
    cif.write(out_path)

def main():
    """
    Main entry point for batch prediction.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Batch Metal Binding Predictor (HETATM)")
    parser.add_argument("--input_dir", required=True, help="Directory containing input protein structures (.pdb or .cif).")
    parser.add_argument("--output_dir", required=True, help="Directory to save predicted structures with metal binding sites.")
    parser.add_argument("--model", required=True, help="Path to the trained PyTorch model checkpoint (.pth).")
    parser.add_argument("--config", default=FEATURE_CONFIG_PATH, help="Path to the JSON feature configuration file.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size of the model.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Probability threshold for predicting a metal site.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model_safe(args.model, args.hidden_dim)
    feat_config = load_config(args.config)
    
    files = [f for f in os.listdir(args.input_dir) if f.endswith((".cif", ".pdb"))]
    for f in tqdm(files, desc="Batch Processing"):
        try:
            in_path = os.path.join(args.input_dir, f)
            tensors, prot_atoms = prepare_protein_for_prediction(in_path, feat_config)
            
            if tensors is None: continue
            
            data = HeteroData()
            data['protein'].x = tensors['atom_x'].to(DEVICE)
            data['protein'].pos = tensors['atom_pos_heavy'].to(DEVICE)
            data['protein'].batch = torch.zeros(len(tensors['atom_x']), dtype=torch.long).to(DEVICE)
            
            preds = predict_structure(model, data, args.threshold)
            
            preds = filter_overlapping_predictions(preds, distance_cutoff=2.0)
            
            if len(preds) == 0:
                print(f"\n[Info] No metals predicted for {f} above threshold.")
                continue
            
            out_name = os.path.splitext(f)[0] + "_predicted.cif"
            write_results(prot_atoms, preds, os.path.join(args.output_dir, out_name))
            
        except Exception as e:
            print(f"\n[Error] Failed to process {f}: {e}")

if __name__ == "__main__":
    main()
