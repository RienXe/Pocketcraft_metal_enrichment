"""Module defining the PyTorch Geometric dataset for protein-metal binding prediction."""

import os
import torch
from torch_geometric.data import HeteroData, Batch
from utils.config import METAL_VOCABULARY, METAL_TO_IDX

class ProteinUniversalMetalDataset(torch.utils.data.Dataset):
    """
    Dataset for metal binding prediction.

    This dataset loads preprocessed protein graph features and target metal
    binding sites from saved PyTorch dictionary files.

    Attributes:
        root_dir (str): Directory containing the preprocessed files.
        max_atoms (int or None): Maximum number of atoms permitted in a protein.
        augment (bool): Whether to apply random 3D rotations to coordinates.
        representation (str): The geometric representation type ('atom' or 'surface').
        files (list of str): List of valid preprocessed files in the root directory.
    """
    def __init__(self, root_dir, files=None, max_atoms=None, augment=False, representation='atom'):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Directory where the preprocessed PyTorch files are located.
            files (list of str, optional): Specific list of filenames to use. If None, all `.pt` files in `root_dir` are used.
            max_atoms (int, optional): Skip structures with more atoms than this limit. Defaults to None.
            augment (bool, optional): Apply random rotational augmentation during loading. Defaults to False.
            representation (str, optional): The graph representation format ('atom' or 'surface'). Defaults to 'atom'.

        Raises:
            ValueError: If the representation is not 'atom' or 'surface'.
        """
        self.root_dir = root_dir
        self.max_atoms = max_atoms
        self.augment = augment
        if representation not in ('atom', 'surface'):
            raise ValueError(f"representation must be 'atom' or 'surface', got {representation!r}")
        self.representation = representation
        if files is None:
            self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])
        else:
            self.files = [f for f in files if os.path.exists(os.path.join(root_dir, f))]

    def __len__(self):
        """
        Get the total number of files in the dataset.

        Returns:
            int: The number of files.
        """
        return len(self.files)

    @staticmethod
    def _random_rotation():
        """
        Generate a random 3D rotation matrix.

        Returns:
            torch.Tensor: A 3x3 orthogonal rotation matrix.
        """
        A = torch.randn(3, 3)
        Q, R = torch.linalg.qr(A)
        d = torch.sign(torch.diagonal(R))
        Q = Q * d.unsqueeze(0)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    def __getitem__(self, idx):
        """
        Load and return a graph representation of a protein and its target metals.

        Args:
            idx (int): Index of the file to load.

        Returns:
            list of HeteroData or None: A list of heterogeneous graph data objects,
            one for each unique metal type bound to the protein. Returns None if
            the structure is invalid or exceeds the maximum atom limit.
        """
        path = os.path.join(self.root_dir, self.files[idx])
        try:
            d = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            return None

        if 'ligand_features' not in d:
            return None

        if self.representation == 'atom' and 'atom_x' in d and 'atom_pos_heavy' in d:
            node_x   = d['atom_x']
            node_pos = d['atom_pos_heavy']
        elif self.representation == 'atom' and 'atom_x' in d and 'atom_pos' in d:
            node_x   = d['atom_x']
            node_pos = d['atom_pos']
        elif 'x' in d and 'pos' in d:
            node_x   = d['x']
            node_pos = d['pos']
        else:
            return None

        if self.max_atoms is not None and node_x.size(0) > self.max_atoms:
            return None

        metal_dict = {}
        # Parse the metal targets out of ligand_features block
        for ccd_id, parts in d['ligand_features'].items():
            # In v4 scripts, ccd_id might be instance_id (e.g. "ATP_A_100").
            # But the dataset only handles original string forms correctly if we fall back to entry details
            if isinstance(parts, dict) and 'ccd_id' in parts:
                ccd_str = str(parts['ccd_id']).upper()
            else:
                ccd_str = str(ccd_id).upper()
                
            if ccd_str in METAL_VOCABULARY:
                # If parts is actually a dict containing 'parts' (v4 format):
                ligand_parts = parts.get('parts', parts) if isinstance(parts, dict) else parts
                if isinstance(ligand_parts, dict):
                    for _pid, lig in ligand_parts.items():
                        if isinstance(lig, dict) and 'x' in lig and lig['x'].size(0) == 1:
                            metal_dict.setdefault(ccd_str, []).append(lig['atom_pos_structure'][0])

        if not metal_dict:
            return None

        protein_pos = node_pos
        if self.augment:
            R = self._random_rotation()
            centroid = protein_pos.mean(dim=0, keepdim=True)
            protein_pos = (protein_pos - centroid) @ R.T + centroid
            for k in metal_dict:
                metal_dict[k] = [(p.unsqueeze(0) - centroid) @ R.T + centroid
                                 for p in metal_dict[k]]
                metal_dict[k] = [p.squeeze(0) for p in metal_dict[k]]

        graphs = []
        for m_type, coords in metal_dict.items():
            data = HeteroData()
            data['protein'].x = node_x
            data['protein'].pos = protein_pos
            data.true_metal_pos = torch.stack(coords)
            data.num_metals = torch.tensor([len(coords)], dtype=torch.long)
            data.metal_type = torch.tensor([METAL_TO_IDX[m_type]], dtype=torch.long)
            graphs.append(data)

        return graphs

def collate_flatten_skip_none(batch):
    """
    Collate function to filter out None values and flatten lists of graphs.

    Args:
        batch (list): A list of items returned by the dataset's __getitem__.

    Returns:
        Batch or None: A PyTorch Geometric Batch object containing all valid
        graphs from the batch, or None if no valid graphs were found.
    """
    flat = []
    for sublist in batch:
        if sublist is not None:
            flat.extend(sublist)
    if not flat:
        return None
    return Batch.from_data_list(flat)
