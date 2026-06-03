"""Module for visualizing training histories, ROC curves, and prediction histograms."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_history(history, save_path):
    """
    Plot the training and validation history.

    Args:
        history (list of dict): List containing epoch-wise training statistics.
        save_path (str): Path to save the generated PNG plot.

    Returns:
        None
    """
    if not history:
        return
    epochs       = [r['epoch'] for r in history]
    train_loss   = [r['train_loss'] for r in history]
    train_dist   = [r.get('train_mean_dist', float('nan')) for r in history]
    val_dist     = [r['val_mean_dist'] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(epochs, train_loss, label='Train Loss', color='#1565C0', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(epochs, val_dist, color='#AD1457', lw=2, label='Val Mean Dist')
    ax.plot(epochs, train_dist, color='#1565C0', lw=2, linestyle='--', label='Train Mean Dist')
    if val_dist:
        i_best = int(np.nanargmin(val_dist))
        ax.scatter([epochs[i_best]], [val_dist[i_best]], s=70, zorder=5,
                   color='#AD1457', edgecolors='white', linewidths=1.0,
                   label=f'best {val_dist[i_best]:.2f} Å @ ep {epochs[i_best]}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean distance error (Å)')
    ax.set_title('Distance Error Over Time', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.25)

    ax = axes[2]
    if 'val_success_5A' in history[0]:
        s1 = [r.get('val_success_1A', float('nan')) for r in history]
        s2 = [r.get('val_success_2A', float('nan')) for r in history]
        s5 = [r.get('val_success_5A', float('nan')) for r in history]
        ax.plot(epochs, s1, label='<1 Å', color='#2E7D32', lw=2)
        ax.plot(epochs, s2, label='<2 Å', color='#1565C0', lw=2)
        ax.plot(epochs, s5, label='<5 Å', color='#AD1457', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Success rate')
    ax.set_title('Distance-Threshold Success Rates', fontweight='bold')
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_roc_curve(filtered_data, title, out_path):
    """
    Plot Receiver Operating Characteristic (ROC) curves.

    Args:
        filtered_data (dict): Dictionary mapping labels to lists of evaluation result dictionaries.
        title (str): Title of the plot.
        out_path (str): Path to save the generated PNG plot.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    valid_lines = 0
    for label, items in filtered_data.items():
        y_true = [item['y_true'] for item in items]
        y_scores = [item['y_score'] for item in items]
        if sum(y_true) == 0: continue
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc(fpr, tpr):.3f})')
        valid_lines += 1

    if valid_lines > 0:
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_metal_histograms(data_list, target_metals, metric_key, title, xlabel, out_path, x_range, bins=30):
    """
    Plot grid histograms showing metric distribution for each target metal.

    Args:
        data_list (list of dict): Evaluation data points.
        target_metals (list of str): Ordered list of metals to plot.
        metric_key (str): The dictionary key of the metric to plot.
        title (str): Title of the figure.
        xlabel (str): Label for the x-axis.
        out_path (str): Path to save the generated PNG plot.
        x_range (tuple): Range of the x-axis (min, max).
        bins (int, optional): Number of histogram bins. Defaults to 30.

    Returns:
        None
    """
    fig, axes = plt.subplots(3, 4, figsize=(25, 20))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, metal in enumerate(target_metals):
        ax = axes[idx]
        metal_data = [p[metric_key] for p in data_list if p['gt_type'] == metal and p['y_true'] == 1 and p[metric_key] is not None and p['y_score'] > 0]
        
        if len(metal_data) > 0:
            ax.hist(metal_data, bins=bins, range=x_range, color=colors[idx % len(colors)], alpha=0.7, edgecolor='black')
        ax.set_title(f"Metal: {metal} (n={len(metal_data)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.5)

    axes[11].axis('off')
    if len(target_metals) <= 10:
        axes[10].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_metal_category_histograms(data_list, target_metals, metric_key, category, title, xlabel, out_path, x_range, bins=30):
    """
    Plot grid histograms showing metric distribution for each target metal, filtered by category.

    Args:
        data_list (list of dict): Evaluation data points.
        target_metals (list of str): Ordered list of metals to plot.
        metric_key (str): The dictionary key of the metric to plot.
        category (str): The specific category to filter by (e.g., 'functional').
        title (str): Title of the figure.
        xlabel (str): Label for the x-axis.
        out_path (str): Path to save the generated PNG plot.
        x_range (tuple): Range of the x-axis (min, max).
        bins (int, optional): Number of histogram bins. Defaults to 30.

    Returns:
        None
    """
    fig, axes = plt.subplots(3, 4, figsize=(25, 20))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, metal in enumerate(target_metals):
        ax = axes[idx]
        metal_data = [
            p[metric_key] for p in data_list 
            if p['gt_type'] == metal and p['cat'] == category and p['status'] == 'TP' and p[metric_key] is not None
        ]
        
        if len(metal_data) > 0:
            ax.hist(metal_data, bins=bins, range=x_range, color=colors[idx % len(colors)], alpha=0.7, edgecolor='black')
        
        ax.set_title(f"Metal: {metal} (Cat {category}, n={len(metal_data)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.5)

    axes[11].axis('off')
    if len(target_metals) <= 10:
        axes[10].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
