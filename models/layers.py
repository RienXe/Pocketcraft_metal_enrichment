"""Module containing the implementation of Equivariant Graph Neural Network layers."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer updating both features and coordinates.

    This layer computes messages between nodes based on their features and distances,
    updating node embeddings and their corresponding 3D coordinates.

    Attributes:
        edge_mlp (nn.Sequential): Network to compute edge messages.
        coord_mlp (nn.Sequential): Network to compute coordinate updates.
        node_mlp (nn.Sequential): Network to compute node feature updates.
    """
    def __init__(self, hidden_dim, num_rbf, dropout=0.15):
        """
        Initialize the EGNN layer.

        Args:
            hidden_dim (int): Dimensionality of the node and edge embeddings.
            num_rbf (int): Number of radial basis function centers.
            dropout (float, optional): Dropout probability. Defaults to 0.15.
        """
        super().__init__(aggr='mean') 
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, pos, edge_index, rbf):
        """
        Forward pass for the EGNN layer.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, hidden_dim].
            pos (torch.Tensor): Node coordinates of shape [num_nodes, 3].
            edge_index (torch.Tensor): Graph edge indices of shape [2, num_edges].
            rbf (torch.Tensor): Radial basis function embeddings of distances [num_edges, num_rbf].

        Returns:
            tuple: A tuple containing:
                - x_new (torch.Tensor): Updated node features.
                - pos_new (torch.Tensor): Updated node coordinates.
        """
        row, col = edge_index
        
        edge_feat = torch.cat([x[row], x[col], rbf], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        
        # BOUNDED coordinate update
        coord_weight = torch.tanh(self.coord_mlp(m_ij)) 
        direction = pos[row] - pos[col]
        pos_update = direction * coord_weight
        
        # Aggregate coordinate updates and normalize by degree to prevent explosion
        from torch_scatter import scatter_add
        pos_aggr = scatter_add(pos_update, row, dim=0, dim_size=pos.size(0))
        degree = scatter_add(torch.ones_like(coord_weight), row, dim=0, dim_size=pos.size(0))
        new_pos = pos + (pos_aggr / degree.clamp(min=1.0))
        
        m_i = self.propagate(edge_index, m_ij=m_ij)
        node_update = self.node_mlp(torch.cat([x, m_i], dim=-1))
        
        return x + node_update, new_pos

    def message(self, m_ij):
        """
        Return the pre-computed messages.

        Args:
            m_ij (torch.Tensor): Edge messages.

        Returns:
            torch.Tensor: The messages to aggregate.
        """
        return m_ij
