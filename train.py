"""Entry point for training the metal binding prediction model."""

import argparse
from training.trainer import train

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Universal Metal Binding Predictor (anchor + offset)")
    p.add_argument('--dir_train', required=True, help="Directory containing training .pt files")
    p.add_argument('--dir_val', type=str, default=None, 
                   help="Optional: dedicated directory containing .pt files for validation. "
                        "If provided, --cluster_dict split logic is skipped.")
    p.add_argument('--cluster_dict', required=True, help="Path to JSON file mapping sequence clusters.")
    p.add_argument('--out_dir', default='metal_logs', help="Directory to save the trained model and logs.")
    p.add_argument('--pretrained', type=str, default=None, 
                help="Path to a best.pth or latest checkpoint to resume training.")
    p.add_argument('--val_fraction', type=float, default=0.15, help="Fraction of clusters reserved for validation.")
    p.add_argument('--seed', type=int, default=42, help="Random seed for splitting and initialization.")
    p.add_argument('--epochs', type=int, default=100, help="Maximum number of training epochs.")
    p.add_argument('--batch_size', type=int, default=16, help="Batch size for DataLoader.")
    p.add_argument('--lr', type=float, default=3e-4, help="Learning rate for AdamW optimizer.")
    p.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for AdamW optimizer.")
    p.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension for EGNN and linear layers.")
    p.add_argument('--num_layers', type=int, default=4, help="Number of EGNN message passing layers.")
    p.add_argument('--num_rbf', type=int, default=16, help="Number of radial basis functions (RBF) for distance encoding.")
    p.add_argument('--cutoff', type=float, default=8.0, help="Distance cutoff in Angstroms for radius graph construction.")
    p.add_argument('--max_neighbors', type=int, default=32, help="Maximum neighbors allowed per atom in the radius graph.")
    p.add_argument('--dropout', type=float, default=0.1, help="Dropout probability in MLP layers.")
    p.add_argument('--rotate_aug', type=int, default=1, help="Flag (1/0) to enable random SE(3) rotation augmentation during training.")
    p.add_argument('--max_per_cluster', type=int, default=0, help="Maximum number of PDB files sampled per cluster (0 = no limit).")
    p.add_argument('--representation', choices=['atom', 'surface'], default='atom', help="Input tensor representation type ('atom' or 'surface').")
    p.add_argument('--reg_weight', type=float, default=1.0, help="Weight of the offset regression loss relative to the classification loss.")
    p.add_argument('--warmup_epochs', type=int, default=2, help="Number of epochs for learning rate warmup.")
    p.add_argument('--num_workers', type=int, default=12, help="Number of DataLoader worker processes.")
    p.add_argument('--max_atoms', type=int, default=4000, help="Skip protein structures containing more than this number of atoms.")
    p.add_argument('--patience', type=int, default=15, help="Number of epochs to wait for validation improvement before early stopping.")
    p.add_argument('--min_delta', type=float, default=0.001, help="Minimum change in validation distance to qualify as an improvement.")
    
    args = p.parse_args()
    if args.max_atoms == 0:
        args.max_atoms = None
    train(args)
