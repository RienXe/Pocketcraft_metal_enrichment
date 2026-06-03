"""Module providing data loading and dataset splitting utilities for metal binding prediction."""

import json
import random

def get_strict_cluster_split(cluster_dict_path, val_fraction=0.15, seed=42, max_per_cluster=0):
    """
    Generate a strict cluster-based train and validation split.

    This function ensures that structures from the same sequence identity cluster
    do not span across both training and validation sets, preventing data leakage.
    It reads a JSON file mapping cluster IDs to lists of PDB identifiers and
    randomly assigns clusters to either set.

    Args:
        cluster_dict_path (str): Path to the JSON file containing cluster definitions.
        val_fraction (float, optional): The fraction of clusters to reserve for validation. Defaults to 0.15.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        max_per_cluster (int, optional): Maximum number of samples to take per cluster in the training set. Defaults to 0 (no limit).

    Returns:
        tuple: A tuple containing two lists:
            - train_files (list of str): Filenames allocated for the training set.
            - val_files (list of str): Filenames allocated for the validation set.
    """
    with open(cluster_dict_path, 'r') as f:
        cluster_dict = json.load(f)

    metal_clusters = cluster_dict.get('metal', {})
    rng = random.Random(seed)

    cluster_ids = [cid for cid, pdb_list in metal_clusters.items() if pdb_list]
    rng.shuffle(cluster_ids)

    num_val_clusters = max(1, int(len(cluster_ids) * val_fraction))
    val_cluster_ids = cluster_ids[:num_val_clusters]
    train_cluster_ids = cluster_ids[num_val_clusters:]

    train_files, val_files = [], []
    for cid in train_cluster_ids:
        pdbs = list(metal_clusters[cid])
        if max_per_cluster and len(pdbs) > max_per_cluster:
            rng.shuffle(pdbs)
            pdbs = pdbs[:max_per_cluster]
        for pdb in pdbs:
            if not pdb.endswith('.pt'):
                pdb += '.pt'
            train_files.append(pdb)

    for cid in val_cluster_ids:
        chosen_pdb = rng.choice(metal_clusters[cid])
        if not chosen_pdb.endswith('.pt'):
            chosen_pdb += '.pt'
        val_files.append(chosen_pdb)

    print(f"\n--- Cluster Split (30% Seq Identity Cutoff) ---")
    print(f"Total Unique Clusters: {len(cluster_ids)}")
    print(f"  -> Train clusters: {len(train_cluster_ids)} (yielding {len(train_files)} files"
          + (f", capped at {max_per_cluster}/cluster)" if max_per_cluster else ")"))
    print(f"  -> Val clusters:   {len(val_cluster_ids)} (yielding {len(val_files)} files)\n")

    return train_files, val_files
