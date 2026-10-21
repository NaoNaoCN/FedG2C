#!/usr/bin/env python3
"""Visualize pre-training metrics for every federated client.

Usage:
    python visualization/plot_pretrain_metrics.py <metrics_json_path>

The output directory contains raw/weighted loss curves, loss ratios,
gradient norms, health metrics, and expert-diversity plots.
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot only the losses retained in the public training objective.
LOSS_KEYS_PLOT = [
    'topo_recon',
    # 'riem_topo_recon',  # [DISABLED]
    # 'commit',           # [DISABLED]
    # 'align',            # [DISABLED]
    'struct_align',
    # 'riem_proj',        # [DISABLED]
    'curvature_div_loss'
]
GRAD_GROUPS = ['moe_encoder', 'local_encoder', 'gating', 'decoders', 'sem']

LOSS_COLORS = [
    '#e74c3c',  # topo_recon
    '#1abc9c',  # struct_align
    '#2d5dba'   # curvature_div_loss
]
GRAD_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#e67e22', '#9b59b6']


def _sorted_client_ids(data):
    def _key(x):
        part = x.split('_')[-1]
        return int(part) if part.isdigit() else x
    return sorted(data.keys(), key=_key)


def _subplots_for_clients(num_clients, figsize_per=(7, 3.5)):
    ncols = min(num_clients, 3)
    nrows = (num_clients + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        squeeze=False,
    )
    for idx in range(num_clients, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    return fig, axes


def _ax_for(axes, idx):
    ncols = axes.shape[1]
    return axes[idx // ncols][idx % ncols]


def plot_losses(data, loss_field, title, out_path):
    """Plot raw or weighted loss curves for each client."""
    client_ids = _sorted_client_ids(data)
    num_clients = len(client_ids)
    fig, axes = _subplots_for_clients(num_clients)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for i, cid in enumerate(client_ids):
        ax = _ax_for(axes, i)
        d = data[cid]
        rounds = d['rounds']
        losses = d[loss_field]

        for j, key in enumerate(LOSS_KEYS_PLOT):
            vals = losses.get(key, [])
            if vals:
                ax.plot(rounds, vals, label=key, color=LOSS_COLORS[j], linewidth=1.6)

        total = losses.get('total', [])
        if total:
            ax.plot(rounds, total, label='total', color='#555', linewidth=1.2,
                    linestyle='--', alpha=0.7)

        ax.set_title(f'Client {cid}', fontsize=10)
        ax.set_xlabel('Round', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_loss_ratio(data, out_path):
    """Plot the stacked ratio of weighted loss terms for each client."""
    client_ids = _sorted_client_ids(data)
    num_clients = len(client_ids)
    fig, axes = _subplots_for_clients(num_clients)
    fig.suptitle('Weighted Loss Ratio (stacked area)', fontsize=13, fontweight='bold')

    for i, cid in enumerate(client_ids):
        ax = _ax_for(axes, i)
        d = data[cid]
        rounds = np.array(d['rounds'])
        wl = d['weighted_losses']

        mat = np.array([wl.get(k, [0.0] * len(rounds)) for k in LOSS_KEYS_PLOT])
        total = mat.sum(axis=0)
        total = np.where(total == 0, 1.0, total)
        ratio = mat / total

        ax.stackplot(rounds, ratio, labels=LOSS_KEYS_PLOT, colors=LOSS_COLORS, alpha=0.8)
        ax.set_title(f'Client {cid}', fontsize=10)
        ax.set_xlabel('Round', fontsize=9)
        ax.set_ylabel('Ratio', fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_grad_norms(data, out_path):
    """Plot gradient norms for each tracked model component."""
    client_ids = _sorted_client_ids(data)
    num_clients = len(client_ids)
    fig, axes = _subplots_for_clients(num_clients)
    fig.suptitle('Gradient Norms per Module', fontsize=13, fontweight='bold')

    for i, cid in enumerate(client_ids):
        ax = _ax_for(axes, i)
        d = data[cid]
        rounds = d['rounds']
        gn = d['grad_norms']

        for j, g in enumerate(GRAD_GROUPS):
            vals = gn.get(g, [])
            if vals:
                ax.plot(rounds, vals, label=g, color=GRAD_COLORS[j], linewidth=1.6)

        ax.set_title(f'Client {cid}', fontsize=10)
        ax.set_xlabel('Round', fontsize=9)
        ax.set_ylabel('Grad Norm', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_health_metrics(data, out_path):
    """Plot normalized loss-convergence and MoE-gradient health indicators.

    ``topo_drop`` and ``struct_converge`` measure loss reduction from the
    initial round. ``grad_norm_moe`` is the normalized MoE gradient norm.
    """
    client_ids = _sorted_client_ids(data)
    num_clients = len(client_ids)
    fig, axes = _subplots_for_clients(num_clients)
    fig.suptitle('Pretraining Health Metrics (normalized)', fontsize=13, fontweight='bold')

    for i, cid in enumerate(client_ids):
        ax = _ax_for(axes, i)
        d = data[cid]
        rounds = np.array(d['rounds'])

        topo = np.array(d['raw_losses'].get('topo_recon', []))
        struct = np.array(d['raw_losses'].get('struct_align', []))
        grad_moe = np.array(d['grad_norms'].get('moe_encoder', []))

        if len(topo) > 0 and topo[0] > 1e-8:
            topo_drop = (topo[0] - topo) / topo[0]
            ax.plot(rounds, topo_drop, label='topo_drop', color='#2ecc71', linewidth=1.8)

        if len(struct) > 0 and struct[0] > 1e-8:
            struct_conv = (struct[0] - struct) / struct[0]
            ax.plot(rounds, struct_conv, label='struct_converge', color='#3498db', linewidth=1.8)

        if len(grad_moe) > 0:
            peak = max(grad_moe) + 1e-8
            grad_norm = grad_moe / peak
            ax.plot(rounds, grad_norm, label='grad_moe (norm)', color='#e74c3c',
                    linewidth=1.4, linestyle='--', alpha=0.8)

        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
        ax.set_title(f'Client {cid}', fontsize=10)
        ax.set_xlabel('Round', fontsize=9)
        ax.set_ylabel('Normalized Score', fontsize=9)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_expert_diversity(data, out_path):
    """Plot curvature spread and normalized expert-routing entropy.

    Higher curvature spread indicates more distinct geometries, while routing
    entropy close to one indicates balanced expert usage.
    """
    client_ids = _sorted_client_ids(data)
    num_clients = len(client_ids)
    fig, axes = _subplots_for_clients(num_clients)
    fig.suptitle('Expert Diversity Metrics', fontsize=13, fontweight='bold')

    for i, cid in enumerate(client_ids):
        ax = _ax_for(axes, i)
        d = data[cid]
        rounds = d['rounds']
        div = d.get('diversity', {})

        curv_std = div.get('curvature_std', [])
        entropy = div.get('expert_entropy', [])

        ax2 = ax.twinx()

        if curv_std:
            ax.plot(rounds, curv_std, label='curvature_std', color='#9b59b6', linewidth=1.8)
        if entropy:
            ax2.plot(rounds, entropy, label='expert_entropy', color='#e67e22',
                     linewidth=1.8, linestyle='--')

        ax.axhline(y=0.3, color='#9b59b6', linestyle=':', alpha=0.5, linewidth=1.0)
        ax2.axhline(y=0.9, color='#e67e22', linestyle=':', alpha=0.5, linewidth=1.0)

        ax.set_title(f'Client {cid}', fontsize=10)
        ax.set_xlabel('Round', fontsize=9)
        ax.set_ylabel('Curvature Std', fontsize=9, color='#9b59b6')
        ax2.set_ylabel('Entropy (normalized)', fontsize=9, color='#e67e22')
        ax2.set_ylim(0, 1.1)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_pretrain_metrics.py <metrics_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    with open(json_path) as f:
        data = json.load(f)

    stem = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.join(os.path.dirname(json_path), stem)
    os.makedirs(out_dir, exist_ok=True)

    plot_losses(data, 'raw_losses',
                'Raw Losses (unweighted)',
                os.path.join(out_dir, 'raw_losses.png'))

    plot_losses(data, 'weighted_losses',
                'Weighted Losses (×lambda)',
                os.path.join(out_dir, 'weighted_losses.png'))

    plot_loss_ratio(data, os.path.join(out_dir, 'loss_ratio.png'))

    plot_grad_norms(data, os.path.join(out_dir, 'grad_norms.png'))

    plot_health_metrics(data, os.path.join(out_dir, 'health_metrics.png'))
    plot_expert_diversity(data, os.path.join(out_dir, 'expert_diversity.png'))


if __name__ == '__main__':
    main()
