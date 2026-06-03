"""Module for training the model."""

import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.config import METAL_VOCABULARY
from data.dataset import ProteinUniversalMetalDataset, collate_flatten_skip_none
from data.dataloader import get_strict_cluster_split
from models.predictor import MetalBindingPredictor
from training.loss import selection_offset_loss
from utils.plotting import plot_history

def train(args):
    """
    Execute the training loop for the metal binding predictor.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.dir_val and os.path.isdir(args.dir_val):
        print(f"Using dedicated validation directory: {args.dir_val}")
        train_files = None 
        val_files = None
        val_root = args.dir_val
    else:
        print(f"Using cluster-based split from: {args.dir_train}")
        train_files, val_files = get_strict_cluster_split(
            cluster_dict_path=args.cluster_dict,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_per_cluster=args.max_per_cluster,
        )
        val_root = args.dir_train

    print("Initializing DataLoaders...")
    train_dataset = ProteinUniversalMetalDataset(args.dir_train, files=train_files,
                                                 max_atoms=args.max_atoms, augment=args.rotate_aug,
                                                 representation=args.representation)
    
    val_dataset   = ProteinUniversalMetalDataset(val_root, files=val_files,
                                                 max_atoms=args.max_atoms, augment=False,
                                                 representation=args.representation)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_flatten_skip_none,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    
    model = MetalBindingPredictor(
        protein_in=32,
        hidden_dim=args.hidden_dim,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        num_layers=args.num_layers,
        max_neighbors=args.max_neighbors,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = max(1, args.warmup_epochs * max(1, len(train_loader)))
    total_steps = max(2, args.epochs * max(1, len(train_loader)))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_dist = float('inf')
    epochs_since_best = 0
    peak_atoms_in_batch = 0
    history = []
    start_epoch = 0
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained model from {args.pretrained}...")
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_val_dist = checkpoint.get('best_dist', float('inf'))
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            
        print(f"Resuming from epoch {start_epoch} with best_dist {best_val_dist:.3f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total, train_cls, train_reg, train_dists, train_types, train_anchor_hits = [], [], [], [], [], []

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]", leave=False)
        for batch in bar:
            if batch is None:
                continue
            batch = batch.to(device)
            peak_atoms_in_batch = max(peak_atoms_in_batch, int(batch['protein'].x.size(0)))
            optimizer.zero_grad()

            logits, offsets, updated_pos, p_b = model(batch)

            total, lc, lr_, dists, types, val_anchor_hits = selection_offset_loss(
                logits=logits, 
                offsets=offsets,
                updated_pos=updated_pos, 
                p_pos_original=batch['protein'].pos,
                p_b=p_b, 
                batch=batch, 
                reg_weight=args.reg_weight
            )
            if not dists:
                continue

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_total.append(total.item())
            train_cls.append(float(lc))
            train_reg.append(float(lr_))
            train_dists.extend(dists)
            train_types.extend(types)
            train_anchor_hits.extend(val_anchor_hits)
            bar.set_postfix(
                Loss=f"{total.item():.3f}",
                CE=f"{float(lc):.3f}",
                REG=f"{float(lr_):.3f}",
                Dist=f"{np.mean(dists):.2f}Å",
            )

        avg_train_loss = float(np.mean(train_total)) if train_total else 0.0
        avg_train_cls  = float(np.mean(train_cls))   if train_cls   else 0.0
        avg_train_reg  = float(np.mean(train_reg))   if train_reg   else 0.0
        avg_train_dist = float(np.mean(train_dists)) if train_dists else float('nan')

        model.eval()
        val_dists, val_types, val_anchor_hits = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Val  ]", leave=False):
                if batch is None:
                    continue
                batch = batch.to(device)
                logits, offsets, updated_pos, p_b = model(batch)

                total, lc, lr_, dists, types, val_anchor_hits = selection_offset_loss(
                    logits=logits, 
                    offsets=offsets,
                    updated_pos=updated_pos, 
                    p_pos_original=batch['protein'].pos,
                    p_b=p_b, 
                    batch=batch, 
                    reg_weight=args.reg_weight
                )
                val_dists.extend(dists)
                val_types.extend(types)
                val_anchor_hits.extend(val_anchor_hits)

        if val_dists:
            arr = np.array(val_dists)
            avg_val_dist = float(arr.mean())
            median_dist = float(np.median(arr))
            
            p75 = float(np.percentile(arr, 75))
            p90 = float(np.percentile(arr, 90))
            
            succ_1a = float((arr < 1.0).mean())
            succ_2a = float((arr < 2.0).mean())
            succ_5a = float((arr < 5.0).mean())
            
            anchor_acc = float(np.mean(val_anchor_hits)) if 'val_anchor_hits' in locals() and val_anchor_hits else 0.0

            types_arr = np.array(val_types)
            per_type_stats = []
            for m_idx in np.unique(types_arr):
                mask = types_arr == m_idx
                per_type_stats.append((METAL_VOCABULARY[int(m_idx)],
                                       float(arr[mask].mean()), int(mask.sum())))
        else:
            avg_val_dist = median_dist = p75 = p90 = succ_1a = succ_2a = succ_5a = anchor_acc = float('nan')
            per_type_stats = []

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"[Ep {epoch+1:03d}] LR={cur_lr:.2e} | Train tot {avg_train_loss:.3f} (CE {avg_train_cls:.3f}, REG {avg_train_reg:.3f})")
        print(f"           Val Dist: Mean={avg_val_dist:.2f}Å | Median={median_dist:.2f}Å | 75th={p75:.2f}Å | 90th={p90:.2f}Å")
        print(f"           Success : <1Å={succ_1a:.2f} | <2Å={succ_2a:.2f} | <5Å={succ_5a:.2f} | AnchorHit={anchor_acc:.2f}")
        
        if per_type_stats:
            sorted_types = sorted(per_type_stats, key=lambda x: -x[2])
            type_str = ", ".join(f"{m}={d:.1f}Å(n={n})" for m, d, n in sorted_types)
            print(f"           Per-type: {type_str}")

        history.append({
            'epoch': epoch + 1,
            'lr': cur_lr,
            'train_loss': round(avg_train_loss, 6),
            'train_cls': round(avg_train_cls, 6),
            'train_reg': round(avg_train_reg, 6),
            'train_mean_dist': round(avg_train_dist, 4),
            'val_mean_dist': round(avg_val_dist, 4),
            'val_median_dist': round(median_dist, 4),
            'val_success_1A': round(succ_1a, 4),
            'val_success_2A': round(succ_2a, 4),
            'val_success_5A': round(succ_5a, 4),
        })

        csv_path = os.path.join(args.out_dir, 'history.csv')
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

        plot_history(history, os.path.join(args.out_dir, 'history.png'))

        improved = (not np.isnan(avg_val_dist)) and (avg_val_dist < best_val_dist - args.min_delta)
        if improved:
            best_val_dist = avg_val_dist
            epochs_since_best = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dist': best_val_dist,
                'history': history,
                'args': vars(args),
            }, os.path.join(args.out_dir, 'best.pth'))
            print(f"           ✓ New best meanD {best_val_dist:.3f} Å. Model saved.")
        else:
            epochs_since_best += 1
            if args.patience > 0:
                print(f"           No improvement for {epochs_since_best}/{args.patience} epochs "
                      f"(best={best_val_dist:.3f} Å)")

        if args.patience > 0 and epochs_since_best >= args.patience:
            print(f"\n[!] Early stopping at epoch {epoch+1}: "
                  f"no improvement > {args.min_delta} Å for {args.patience} epochs. "
                  f"Best val meanD = {best_val_dist:.3f} Å.")
            break
