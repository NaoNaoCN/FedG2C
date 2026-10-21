#!/usr/bin/env python3
"""Create t-SNE plots from best-epoch node embeddings.

Each checkpoint produces views for raw features, local/shared/fused embeddings,
and logits. Points are colored by class over all nodes.

Usage:
    python visualization/plot_tsne.py logs/finetune/embeddings/<ts>/client0_disease_<ts>_run0.pt

    python visualization/plot_tsne.py --dir logs/finetune/embeddings/<ts>/

Options:
    --perplexity <int>   t-SNE perplexity (default: 30)
    --max_points <int>   Random-subsampling limit (default: 5000)
    --seed <int>         Sampling and t-SNE seed (default: 42)
"""

import argparse
import os
import os.path as osp
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Order views from raw inputs through latent embeddings to classifier outputs.
VIEW_KEYS = ["raw_x", "z_local", "z_shared", "z_fusion", "logits"]
VIEW_TITLES = {
    "raw_x": "Raw Feature",
    "z_local": "Local Encoder",
    "z_shared": "Shared MoE",
    "z_fusion": "Fusion",
    "logits": "Logits",
}
PCA_THRESHOLD = 50

# Fixed palette for the first ten classes.
PALETTE = [
    "#3b7fbf",
    "#d44a4a",
    "#4caf50",
    "#ff9800",
    "#9c27b0",
    "#00bcd4",
    "#795548",
    "#e91e63",
    "#607d8b",
    "#ffc107",
]


def _color_for(c, n_classes):
    if n_classes <= len(PALETTE):
        return PALETTE[int(c) % len(PALETTE)]
    cmap = plt.get_cmap("tab20")
    return cmap(int(c) % cmap.N)


def _subsample(idx, max_points, rng):
    if len(idx) <= max_points:
        return idx
    chosen = rng.choice(len(idx), size=max_points, replace=False)
    return idx[chosen]


def _plot_one(z, labels, title, save_path, perplexity, seed):
    n_samples, n_dim = z.shape
    if n_dim > PCA_THRESHOLD:
        z = PCA(n_components=PCA_THRESHOLD, random_state=seed).fit_transform(z)
    perp = min(perplexity, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1000,
    )
    z2 = tsne.fit_transform(z)

    classes = np.unique(labels)
    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for c in classes:
        m = labels == c
        ax.scatter(
            z2[m, 0], z2[m, 1],
            s=36,
            alpha=0.75,
            color=_color_for(c, n_classes),
            edgecolors="white",
            linewidths=0.4,
            label=f"class {int(c)}",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        markerscale=1.2,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_one_pt(pt_path, perplexity, max_points, seed):
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    labels_full = payload["labels"].numpy()
    n = labels_full.shape[0]

    rng = np.random.default_rng(seed)
    idx = _subsample(np.arange(n), max_points, rng)
    labels = labels_full[idx]

    pt_stem = osp.splitext(osp.basename(pt_path))[0]
    out_dir = osp.join(osp.dirname(pt_path), pt_stem)
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = payload.get("dataset_name", "")
    run_id = payload.get("run_id", "?")
    k_shot = payload.get("k_shot", -1)
    best_val = payload.get("best_val", None)
    best_test = payload.get("best_test", None)
    base_suffix = f"split={run_id} | k={k_shot}"

    saved = []
    for key in VIEW_KEYS:
        z = payload.get(key, None)
        if z is None:
            continue
        z_np = z.numpy()[idx]
        view_title = VIEW_TITLES.get(key, key)
        suffix = base_suffix
        if key == "logits" and best_val is not None and best_test is not None:
            suffix += f" | val={best_val:.2f} | test={best_test:.2f}"
        title = f"{dataset_name.upper()} {view_title} t-SNE | {suffix}"
        save_path = osp.join(out_dir, f"{key}.png")
        _plot_one(z_np, labels, title, save_path, perplexity, seed)
        saved.append(save_path)

    print(f"[OK] {pt_path}")
    for p in saved:
        print(f"     -> {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None,
                        help="Single embedding checkpoint; mutually exclusive with --dir.")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing embedding checkpoints.")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.path is None and args.dir is None:
        parser.error("Specify either path or --dir.")

    if args.dir is not None:
        pt_files = sorted(glob(osp.join(args.dir, "*.pt")))
        if not pt_files:
            print(f"[Warn] No .pt files found under {args.dir}")
            return
        for p in pt_files:
            plot_one_pt(p, args.perplexity, args.max_points, args.seed)
    else:
        plot_one_pt(args.path, args.perplexity, args.max_points, args.seed)


if __name__ == "__main__":
    main()
