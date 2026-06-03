"""Module containing the loss function for the metal binding predictor."""

import torch
import torch.nn.functional as F

def selection_offset_loss(
    logits,
    offsets,
    updated_pos,
    p_pos_original,
    p_b,
    batch,
    reg_weight=1.0,
):
    """
    Calculate the combined classification and regression loss using Hungarian matching.

    This function matches predicted sites to ground truth metals and computes
    a classification loss for identifying binding atoms and a regression loss
    for predicting the precise metal coordinates.

    Args:
        logits (torch.Tensor): Classification logits of shape [N].
        offsets (torch.Tensor): Predicted coordinate offsets of shape [N, 3].
        updated_pos (torch.Tensor): Updated atom coordinates from the EGNN of shape [N, 3].
        p_pos_original (torch.Tensor): Original atom coordinates of shape [N, 3].
        p_b (torch.Tensor): Batch indices for atoms of shape [N].
        batch (HeteroData or Batch): The input batch object containing ground truth labels.
        reg_weight (float, optional): Weight for the regression loss. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - total (torch.Tensor): The combined loss.
            - loss_cls (torch.Tensor): The classification loss.
            - loss_reg (torch.Tensor): The regression loss.
            - dists (list of float): Distances between predicted and true metals.
            - types (list of int): Integer identifiers of metal types.
            - anchor_hits (list of bool): True if top-1 predicted anchor is close to a metal.
    """
    device = updated_pos.device
    B = int(p_b.max().item()) + 1

    true_b = torch.arange(B, device=device).repeat_interleave(batch.num_metals)
    all_metals = batch.true_metal_pos

    losses_cls, losses_reg = [], []
    dists, types, anchor_hits = [], [], []

    for i in range(B):
        atom_mask = (p_b == i)

        atom_pos_upd = updated_pos[atom_mask]           # [N, 3]
        atom_pos_orig = p_pos_original[atom_mask]       # [N, 3]
        atom_log = logits[atom_mask]                    # [N]
        atom_offsets = offsets[atom_mask]               # [N, 3]

        m = all_metals[true_b == i]                     # [M, 3]
        if m.numel() == 0:
            continue

        # -------------------------------------------------
        # 🧠 Hungarian Matching (GLOBAL assignment)
        # -------------------------------------------------
        all_pred_pos = atom_pos_upd + atom_offsets      # [N, 3]
        dist_matrix = torch.cdist(m, all_pred_pos)      # [M, N]

        cost = dist_matrix.detach().cpu().numpy()
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_atoms = torch.tensor(col_ind, device=device)  # [M]

        # -------------------------------------------------
        # 🎯 Classification Loss (Cross-Entropy)
        # -------------------------------------------------
        target = matched_atoms  # [M]

        loss_cls = F.cross_entropy(
            atom_log.unsqueeze(0).repeat(len(target), 1),  # [M, N]
            target
        )

        # -------------------------------------------------
        # 📍 Regression Loss (offset prediction)
        # -------------------------------------------------
        anchor_pos = atom_pos_upd[matched_atoms]         # [M, 3]
        anchor_offsets = atom_offsets[matched_atoms]     # [M, 3]

        pred_metal = anchor_pos + anchor_offsets         # [M, 3]

        loss_reg_main = F.smooth_l1_loss(pred_metal, m)

        # 🔥 Offset magnitude regularization
        loss_offset_reg = anchor_offsets.norm(dim=-1).mean()

        loss_reg = loss_reg_main + 0.1 * loss_offset_reg

        # -------------------------------------------------
        # 📊 Evaluation (Hungarian, consistent with training)
        # -------------------------------------------------
        with torch.no_grad():
            matched_dists = dist_matrix[row_ind, col_ind]
            dists.extend(matched_dists.tolist())

            # Anchor accuracy (top-1 anchor sanity check)
            top1_idx = atom_log.argmax()
            top1_pos = atom_pos_orig[top1_idx].unsqueeze(0)
            dist_top1 = torch.cdist(top1_pos, m).min().item()
            anchor_hits.append(dist_top1 < 3.5)

        losses_cls.append(loss_cls)
        losses_reg.append(loss_reg)
        types.extend([int(batch.metal_type[i].item())] * m.size(0))

    if not losses_cls:
        zero = torch.tensor(0.0, requires_grad=True, device=device)
        return zero, zero, zero, [], [], []

    loss_cls = torch.stack(losses_cls).mean()
    loss_reg = torch.stack(losses_reg).mean()
    total = loss_cls + reg_weight * loss_reg

    return total, loss_cls.detach(), loss_reg.detach(), dists, types, anchor_hits
