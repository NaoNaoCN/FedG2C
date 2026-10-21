#!/usr/bin/env python3
"""Visualize fine-tuning metrics from JSON logs and aggregate summaries.

Usage:
    python visualization/plot_finetune_metrics.py <metrics_json_path>
    python visualization/plot_finetune_metrics.py --dir logs/finetune/

The output directory includes accuracy and loss curves, summary comparisons,
and dataset-level box plots.
"""

import sys
import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def _aggregate_runs(data, key, transform=None):
    """Align repeated-run series and return epochs, mean, and standard deviation."""
    all_epochs = []
    all_series = []
    for run_data in data.values():
        epochs = run_data.get("epochs", [])
        series = run_data.get(key, [])
        if not epochs or not series:
            continue
        if transform:
            series = [transform(v) for v in series]
        else:
            series = [float(v) for v in series]
        all_epochs.append(epochs)
        all_series.append(series)

    if not all_series:
        return [], [], []

    # Align all repeated runs to their shortest common sequence.
    min_len = min(len(s) for s in all_series)
    common_epochs = all_epochs[0][:min_len]
    arr = np.array([s[:min_len] for s in all_series])
    return common_epochs, arr.mean(axis=0), arr.std(axis=0)


def _plot_mean_std(ax, epochs, mean, std, color, label, linestyle='-', linewidth=2.0):
    ax.plot(epochs, mean, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    ax.fill_between(epochs,
                    np.array(mean) - np.array(std),
                    np.array(mean) + np.array(std),
                    color=color, alpha=0.15)


def plot_metrics_from_json(json_path, out_dir):
    """Plot epoch-level metrics aggregated as mean plus or minus standard deviation."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        print(f"No data found in {json_path}")
        return

    num_runs = len(data)

    # Plot train, validation, and test accuracy.
    fig, ax = plt.subplots(figsize=(10, 5))

    ep_tr, m_tr, s_tr = _aggregate_runs(data, "train")
    ep_va, m_va, s_va = _aggregate_runs(data, "val")
    ep_te, m_te, s_te = _aggregate_runs(data, "test")

    if ep_tr:
        _plot_mean_std(ax, ep_tr, m_tr, s_tr, '#e74c3c', 'Train', linestyle='-')
    if ep_va:
        _plot_mean_std(ax, ep_va, m_va, s_va, '#3498db', 'Val', linestyle='--')
    if ep_te:
        _plot_mean_std(ax, ep_te, m_te, s_te, '#2ecc71', 'Test', linestyle=':')

    # Mark the mean best-validation epoch across runs.
    best_epochs = [v.get("best_epoch") for v in data.values() if v.get("best_epoch") is not None]
    if best_epochs and ep_va:
        avg_best = int(round(np.mean(best_epochs)))
        ax.axvline(x=avg_best, color='gray', linestyle='--', alpha=0.5, linewidth=1,
                   label=f'Avg best epoch ({avg_best})')

    title_suffix = f' ({num_runs} runs, mean±std)' if num_runs > 1 else ''
    ax.set_title(f'Train / Val / Test Accuracy{title_suffix}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'accuracy_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Plot training loss.
    fig, ax = plt.subplots(figsize=(10, 5))

    ep_l, m_l, s_l = _aggregate_runs(data, "loss")
    if ep_l:
        _plot_mean_std(ax, ep_l, m_l, s_l, '#e74c3c', 'Loss')

    title_suffix = f' ({num_runs} runs, mean±std)' if num_runs > 1 else ''
    ax.set_title(f'Training Loss{title_suffix}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def parse_results_summary(summary_path):
    """Parse the fine-tuning results summary log."""
    results = []

    if not os.path.exists(summary_path):
        return results

    with open(summary_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_entry = {}
    for line in lines:
        line = line.strip()
        if line.startswith('==='):
            if current_entry:
                results.append(current_entry)
            current_entry = {'timestamp': line.strip('= ')}
        elif ':' in line and current_entry:
            if 'Seed' in line or 'K-shot' in line:
                parts = line.split(',')
                for part in parts:
                    if ':' in part:
                        key, val = part.split(':', 1)
                        current_entry[key.strip()] = val.strip()
            else:
                # Dataset results
                dataset, score = line.split(':', 1)
                current_entry[dataset.strip()] = score.strip()

    if current_entry:
        results.append(current_entry)

    return results


def plot_summary_comparison(summary_path, out_dir):
    """Plot dataset performance over multiple recorded runs."""
    results = parse_results_summary(summary_path)

    if not results:
        print(f"No results found in {summary_path}")
        return

    # Group result records by dataset.
    datasets = {}
    for entry in results:
        for key, val in entry.items():
            if key not in ['timestamp', 'Seed', 'K-shot'] and '±' in val:
                if key not in datasets:
                    datasets[key] = []
                mean, std = val.split('±')
                datasets[key].append({
                    'mean': float(mean.strip()),
                    'std': float(std.strip()),
                    'timestamp': entry.get('timestamp', 'Unknown')
                })

    if not datasets:
        print("No dataset results to plot")
        return

    # Plot one performance trend per dataset.
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, max(4 * len(datasets), 6)),
                             constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for idx, (dataset, runs) in enumerate(datasets.items()):
        ax = axes[idx]
        x = list(range(len(runs)))
        means = [r['mean'] for r in runs]
        stds = [r['std'] for r in runs]

        ax.errorbar(x, means, yerr=stds, marker='o', capsize=5, capthick=2,
                   linewidth=2, markersize=8, label=dataset)
        ax.set_title(f'{dataset} Performance Over Runs', fontsize=12, fontweight='bold')
        ax.set_xlabel('Run Index', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=9)

        # Label points with their run dates.
        for i, run in enumerate(runs):
            ax.text(i, means[i] + stds[i] + 0.5, run['timestamp'].split()[0],
                   rotation=45, ha='right', fontsize=7, alpha=0.7)

    out_path = os.path.join(out_dir, 'summary_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_dataset_statistics(log_dir, out_dir):
    """Compute and visualize fine-tuning score distributions by dataset."""
    log_files = list(Path(log_dir).glob('*.log'))

    # Collect final scores by dataset.
    dataset_results = {}

    for log_file in log_files:
        # Extract the dataset name from the run-log filename.
        match = re.search(r'_([a-z\-]+)_\d{8}_\d{6}\.log$', log_file.name)
        if not match:
            continue

        dataset = match.group(1)

        # Parse the final summary entry for this dataset.
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        final_match = re.search(rf'{dataset.upper()}: ([\d.]+) ± ([\d.]+)', content, re.IGNORECASE)
        if final_match:
            mean_acc = float(final_match.group(1))
            std_acc = float(final_match.group(2))

            if dataset not in dataset_results:
                dataset_results[dataset] = []
            dataset_results[dataset].append({'mean': mean_acc, 'std': std_acc})

    if not dataset_results:
        print("No dataset results found in log files")
        return

    # Plot score distributions as box plots.
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = sorted(dataset_results.keys())
    data_to_plot = []
    labels = []

    for dataset in datasets:
        means = [r['mean'] for r in dataset_results[dataset]]
        data_to_plot.append(means)
        labels.append(f"{dataset}\n(n={len(means)})")

    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))

    ax.set_title('Finetune Performance Distribution by Dataset', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'dataset_statistics.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize finetune metrics')
    parser.add_argument('path', nargs='?', help='Path to metrics JSON file or log file')
    parser.add_argument('--dir', help='Directory containing log/json files')
    parser.add_argument('--summary', help='Path to results_summary.log')
    args = parser.parse_args()

    if args.dir:
        # Process every metrics JSON file in the directory.
        log_dir = args.dir
        out_dir = os.path.join(log_dir, 'visualizations')
        os.makedirs(out_dir, exist_ok=True)

        # Plot dataset statistics derived from run logs.
        plot_dataset_statistics(log_dir, out_dir)

        # Plot the aggregate summary comparison.
        summary_path = os.path.join(log_dir, 'results_summary.log')
        if os.path.exists(summary_path):
            plot_summary_comparison(summary_path, out_dir)

        # Process metrics JSON files while skipping generated output directories.
        json_files = list(Path(log_dir).glob('metrics_*.json'))
        for json_file in json_files:
            stem = json_file.stem
            json_out_dir = os.path.join(out_dir, stem)
            marker = os.path.join(json_out_dir, 'loss_curves.png')
            if os.path.exists(marker):
                print(f"Skip (already exists): {stem}")
                continue
            os.makedirs(json_out_dir, exist_ok=True)
            plot_metrics_from_json(str(json_file), json_out_dir)

        print(f"\nAll visualizations saved to: {out_dir}")

    elif args.summary:
        out_dir = os.path.dirname(args.summary) or '.'
        plot_summary_comparison(args.summary, out_dir)

    elif args.path:
        path = args.path
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(os.path.dirname(path), stem)
        os.makedirs(out_dir, exist_ok=True)

        if path.endswith('.json'):
            plot_metrics_from_json(path, out_dir)
        else:
            print("Pass a metrics JSON file (metrics_*.json) or use --dir for directory mode.")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python plot_finetune_metrics.py <metrics_json>")
        print("  python plot_finetune_metrics.py --dir logs/finetune/")
        print("  python plot_finetune_metrics.py --summary logs/finetune/results_summary.log")
        sys.exit(1)


if __name__ == '__main__':
    main()
