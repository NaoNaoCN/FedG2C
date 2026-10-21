#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot fine-tuning efficiency from per-client metrics JSON files.

The script reads stopping epochs, best test scores, runtime, and peak GPU
memory for one or more standard runs. For example:

  python visualization/plot_efficiency_scatter.py logs/finetune/metrics_client0_cora_20260623_143022.json --output efficiency_cora.png
  python visualization/plot_efficiency_scatter.py --dir logs/finetune/ --dataset cora --output efficiency_cora.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


def load_metrics_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Extract efficiency records for all standard runs in one metrics file.

    Returns:
        List of dicts, each containing:
          - stop_epoch: int
          - runtime_sec: float
          - peak_gpu_mb: float
          - best_test: float
          - best_val: float
          - best_epoch: int
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    runs = []
    for run_id_str, run_data in data.items():
        stop_epoch = run_data.get("stop_epoch")
        runtime_sec = run_data.get("runtime_sec")
        peak_gpu_mb = run_data.get("peak_gpu_mb")
        best_test = run_data.get("best_test")
        best_val = run_data.get("best_val")
        best_epoch = run_data.get("best_epoch")

        if stop_epoch is None or best_test is None:
            continue

        runs.append({
            "stop_epoch": stop_epoch,
            "runtime_sec": runtime_sec if runtime_sec is not None else 0.0,
            "peak_gpu_mb": peak_gpu_mb if peak_gpu_mb is not None else 0.0,
            "best_test": best_test,
            "best_val": best_val if best_val is not None else 0.0,
            "best_epoch": best_epoch if best_epoch is not None else stop_epoch,
        })

    return runs


def find_metrics_files(log_dir: str, dataset: str = None, client_id: int = None) -> List[str]:
    """Return metrics JSON files that match optional dataset and client filters.

    Args:
        log_dir: Root directory containing fine-tuning logs.
        dataset: Optional dataset filter.
        client_id: Optional client-ID filter.

    Returns:
        List of file paths
    """
    log_path = Path(log_dir)
    pattern = "metrics_client*.json"
    files = list(log_path.glob(pattern))

    if dataset:
        files = [f for f in files if f"_{dataset.lower()}_" in f.name.lower()]

    if client_id is not None:
        files = [f for f in files if f"client{client_id}_" in f.name.lower()]

    return [str(f) for f in sorted(files)]


def plot_efficiency_scatter(
    runs_dict: Dict[str, List[Dict[str, Any]]],
    output_path: str = "efficiency_scatter.png",
    x_metric: str = "stop_epoch",
    y_metric: str = "best_test",
    color_metric: str = "peak_gpu_mb",
):
    """Plot stopping-time and peak-memory efficiency scatter plots.

    Args:
        runs_dict: Mapping from method names to run records.
        output_path: Destination image path.
        x_metric: Retained CLI compatibility parameter for the horizontal metric.
        y_metric: Retained CLI compatibility parameter for the vertical metric.
        color_metric: Retained CLI compatibility parameter for point color.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax2 = axes[1]

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'x']

    for idx, (method_name, runs) in enumerate(runs_dict.items()):
        if not runs:
            continue

        x1 = [r["stop_epoch"] for r in runs]
        x2 = [r["peak_gpu_mb"] for r in runs]
        y = [r["best_test"] for r in runs]
        c1 = [r["peak_gpu_mb"] for r in runs]
        c2 = [r["stop_epoch"] for r in runs]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        ax1.scatter(x1, y, c=color, marker=marker, s=100, alpha=0.6, edgecolors='k', linewidth=0.5, label=method_name)
        ax2.scatter(x2, y, c=color, marker=marker, s=100, alpha=0.6, edgecolors='k', linewidth=0.5, label=method_name)

        # Mark the mean when a method has repeated runs.
        if len(x1) > 1:
            mean_x1, mean_y1 = np.mean(x1), np.mean(y)
            mean_x2, mean_y2 = np.mean(x2), np.mean(y)
            ax1.scatter(mean_x1, mean_y1, c=color, marker='X', s=300, edgecolors='k', linewidth=2, zorder=10)
            ax2.scatter(mean_x2, mean_y2, c=color, marker='X', s=300, edgecolors='k', linewidth=2, zorder=10)

    ax1.set_xlabel("Time Efficiency (stop_epoch)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Time Efficiency (episodes)", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    ax2.set_xlabel("GPU Memory Efficiency (GB)", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("GPU Memory Efficiency (GB)", fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Plot] Saved efficiency scatter plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot stopping-time and GPU-memory efficiency from metrics JSON files."
    )
    parser.add_argument("files", nargs='*', help="One or more metrics JSON files.")
    parser.add_argument("--dir", type=str, help="Directory containing metrics JSON files.")
    parser.add_argument("--dataset", type=str, help="Optional dataset filter, such as cora.")
    parser.add_argument("--client", type=int, help="Optional client-ID filter.")
    parser.add_argument("--output", type=str, default="efficiency_scatter.png", help="Output image path.")
    parser.add_argument("--x", type=str, default="stop_epoch", choices=["stop_epoch", "runtime_sec"], help="Horizontal metric.")
    parser.add_argument("--y", type=str, default="best_test", choices=["best_test", "best_val"], help="Vertical metric.")
    parser.add_argument("--dump-txt", type=str, default=None,
                        help="Write tab-separated records without generating a plot.")

    args = parser.parse_args()

    files = args.files if args.files else []
    if args.dir:
        files.extend(find_metrics_files(args.dir, dataset=args.dataset, client_id=args.client))

    if not files:
        print("[Error] No metrics JSON files provided. Use --files or --dir to specify input.")
        return

    # Treat each input file as one method labeled by its filename stem.
    runs_dict = {}
    for f in files:
        method_name = Path(f).stem
        runs = load_metrics_from_json(f)
        if runs:
            runs_dict[method_name] = runs
            print(f"[Load] {f}: {len(runs)} runs")

    if not runs_dict:
        print("[Error] No valid data extracted from input files.")
        return

    if args.dump_txt:
        with open(args.dump_txt, 'w', encoding='utf-8') as f:
            f.write("# source_file\trun_id\tstop_epoch\tbest_test(%)\truntime_sec\tpeak_gpu_mb\n")
            for method_name, runs in runs_dict.items():
                for i, r in enumerate(runs):
                    f.write(f"{method_name}\t{i}\t{r['stop_epoch']}\t{r['best_test']}\t{r['runtime_sec']}\t{r['peak_gpu_mb']}\n")
        print(f"[Dump] Saved efficiency data to: {args.dump_txt}")
        return

    plot_efficiency_scatter(
        runs_dict=runs_dict,
        output_path=args.output,
        x_metric=args.x,
        y_metric=args.y,
        color_metric="peak_gpu_mb",
    )


if __name__ == "__main__":
    main()
