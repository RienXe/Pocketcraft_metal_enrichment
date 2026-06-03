"""Module defining the primary model architecture for metal binding prediction."""

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from utils.config import METAL_VOCABULARY
from models.layers import EGNNLayer

class MetalBindingPredictor(nn.Module):
    """
    Equivariant Graph Neural Network model for predicting metal binding sites.

    This model processes a protein structure and a target metal type to predict
    binding probabilities for each atom (or surface point) and regress 3D offsets
    pointing to the precise metal binding location.

    Attributes:
        cutoff (float): Distance cutoff for constructing the radius graph.
        max_neighbors (int): Maximum number of neighbors per node in the graph.
        protein_proj (nn.Sequential): Projection network for initial node features.
        metal_emb (nn.Embedding): Embedding layer for the target metal type.
        rbf_centers (torch.Tensor): Centers for the radial basis functions.
        rbf_width (float): Width of the radial basis functions.
        layers (nn.ModuleList): Sequential list of EGNN layers.
        score_head (nn.Sequential): Network to predict binding probability logits.
        offset_head (nn.Sequential): Network to predict 3D coordinate offsets.
    """
    def __init__(self, protein_in=32, hidden_dim=128, num_rbf=16, cutoff=8.0, num_layers=4, max_neighbors=32, dropout=0.15):
        """
        Initialize the MetalBindingPredictor.

        Args:
            protein_in (int, optional): Input dimensionality of protein features. Defaults to 32.
            hidden_dim (int, optional): Hidden dimensionality of embeddings. Defaults to 128.
            num_rbf (int, optional): Number of radial basis function centers. Defaults to 16.
            cutoff (float, optional): Distance cutoff for the radius graph. Defaults to 8.0.
            num_layers (int, optional): Number of EGNN layers. Defaults to 4.
            max_neighbors (int, optional): Maximum neighbors allowed in the radius graph. Defaults to 32.
            dropout (float, optional): Dropout probability. Defaults to 0.15.
        """
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_in, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.metal_emb = nn.Embedding(len(METAL_VOCABULARY), hidden_dim)
        self.register_buffer('rbf_centers', torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_width = cutoff / max(num_rbf - 1, 1)

        self.layers = nn.ModuleList(
            [EGNNLayer(hidden_dim, num_rbf, dropout) for _ in range(num_layers)]
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()  # keeps offsets bounded
        )

    def expand_distances(self, d):
        """
        Compute radial basis function expansion of distances.

        Args:
            d (torch.Tensor): Pairwise distances of shape [num_edges, 1].

        Returns:
            torch.Tensor: RBF expanded distances of shape [num_edges, num_rbf].
        """
        return torch.exp(-((d - self.rbf_centers) ** 2) / (self.rbf_width ** 2))

    def forward(self, batch):
        """
        Forward pass for the predictor.

        Args:
            batch (HeteroData or Batch): PyTorch Geometric batch object containing
                'protein' node features and coordinates, and 'metal_type'.

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Prediction scores of shape [num_nodes].
                - offsets (torch.Tensor): Predicted coordinate offsets of shape [num_nodes, 3].
                - p_pos (torch.Tensor): Updated node coordinates of shape [num_nodes, 3].
                - p_b (torch.Tensor): Batch indices for the nodes.
        """
        p_x = self.protein_proj(batch['protein'].x)
        p_pos = batch['protein'].pos.clone() 
        p_b = batch['protein'].batch
        
        m_x = self.metal_emb(batch.metal_type.view(-1))[p_b]  
        p_x = p_x + m_x 

        edge_index = radius_graph(p_pos, r=self.cutoff, batch=p_b, max_num_neighbors=self.max_neighbors)
        
        for layer in self.layers:
            d = (p_pos[edge_index[0]] - p_pos[edge_index[1]]).norm(dim=-1, keepdim=True)
            rbf = self.expand_distances(d)
            p_x, p_pos = layer(p_x, p_pos, edge_index, rbf)

        logits = self.score_head(p_x).squeeze(-1)
        offsets = 2.5 * self.offset_head(p_x)
        
        return logits, offsets, p_pos, p_b
