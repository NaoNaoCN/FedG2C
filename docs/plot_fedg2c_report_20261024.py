"""Render the report figures from saved validation evidence; no training is run.

Run from any directory with Python and matplotlib:
    python docs/plot_fedg2c_report_20261024.py
The exported figure data can also be reused with --data PATH.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


ROOT = Path(__file__).resolve().parents[1]
MODES = ("none", "lora", "adapter", "prompt")
COLORS = ("#2463A6", "#D77A25", "#208879", "#9263A6")
PREFIX = "fedg2c-report-20261024"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_data():
    budget_path = "runs/molecular_budget_analysis_20261024/budget_summary.json"
    snapshot_path = "runs/report_evidence_20261024/remote_results_snapshot.json"
    budget = read_json(ROOT / budget_path)
    snapshot = {
        item["relative_path"]: item["content"]
        for item in read_json(ROOT / snapshot_path)["files"]
    }
    original = {}
    for mode in MODES:
        history = budget["target"][mode]["history"]
        selected = min(history, key=lambda row: row["valid"]["loss"])
        original[mode] = {
            "best_epoch": selected["epoch"],
            "epochs_ran": len(history),
            "validation": selected["valid"],
        }
    configs = []
    specs = (
        ("10 轮 · 20 epochs", 10, 20, None, original, budget_path),
        (
            "10 轮 · 最多 50 epochs",
            10, 50, 10,
            snapshot["runs/molecular_target_budget_20261024/summary.json"],
            "runs/molecular_target_budget_20261024/summary.json",
        ),
        (
            "20 轮 · 最多 50 epochs",
            20, 50, 10,
            snapshot["runs/molecular_round20_server10_20261024/summary.json"],
            "runs/molecular_round20_server10_20261024/summary.json",
        ),
    )
    for label, rounds, max_epochs, patience, records, source in specs:
        configs.append({
            "label": label,
            "rounds": rounds,
            "server_steps": 10,
            "max_epochs": max_epochs,
            "patience": patience,
            "source": source,
            "modes": {
                mode: {
                    "best_epoch": records[mode]["best_epoch"],
                    "epochs_ran": records[mode]["epochs_ran"],
                    "validation": {
                        key: records[mode]["validation"][key]
                        for key in ("loss", "roc_auc", "average_precision")
                    },
                }
                for mode in MODES
            },
        })
    return {
        "as_of": "2026-10-24 21:31 Asia/Shanghai (saved evidence)",
        "seed": 2026,
        "selection": "minimum validation BCE; first tie; same epoch for all metrics",
        "sources": [budget_path, snapshot_path],
        "source_training": {
            name: [
                {key: row[key] for key in ("round", "mean_loss")}
                for row in rows
            ]
            for name, rows in budget["source"].items()
        },
        "server_replay": budget["server"],
        "original_validation_history": {
            mode: [
                {"epoch": row["epoch"], "loss": row["valid"]["loss"]}
                for row in budget["target"][mode]["history"]
            ]
            for mode in MODES
        },
        "completed_configs": configs,
    }


def configure_style():
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelcolor": "#334155",
        "text.color": "#1E293B",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.edgecolor": "#CBD5E1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


def save_figure(fig, output, name):
    for suffix in ("png", "svg"):
        path = output / f"{PREFIX}-{name}.{suffix}"
        fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.15)
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    plt.close(fig)


def plot_budget(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 9.6))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.11, hspace=0.60, wspace=0.25)
    fig.suptitle("训练预算审计：分别观察 A、B 和 C", fontsize=19, fontweight="bold", y=0.975)
    fig.text(0.5, 0.927, "10 轮正式运行的已保存产物  |  不同面板回答不同问题", ha="center", color="#64748B")

    ax = axes[0, 0]
    source_names = {"molpcba": "MolPCBA", "proteins": "PROTEINS", "imdb_binary": "IMDB-BINARY"}
    for color, (name, rows) in zip(COLORS, data["source_training"].items()):
        ax.plot([r["round"] for r in rows], [r["mean_loss"] for r in rows],
                color=color, marker="o", markersize=4.5, linewidth=2, label=source_names[name])
    ax.set(title="(a) 模块 A：逐轮本地训练统计", xlabel="通信轮次", ylabel="每客户端每轮平均训练损失")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="upper right", fontsize=9.5)

    rows = data["server_replay"]
    steps = [r["steps"] for r in rows]
    ax = axes[0, 1]
    for key, label, color in (("objective", "校准目标", COLORS[0]),
                              ("diagnostic_output_mse", "独立诊断输出 MSE", COLORS[1])):
        ax.plot(steps, [r[key] for r in rows], marker="o", markersize=4.5,
                color=color, linewidth=2, label=label)
    ax.set(title="(b) 模块 B：固定上传后的独立重放", xlabel="服务器优化步数", ylabel="10 个已保存轮次的均值")
    ax.set_xticks(steps)
    ax.legend(loc="upper right", fontsize=9.5)

    ax = axes[1, 0]
    ax.plot(steps, [r["diagnostic_response_mse"] * 1e6 for r in rows],
            marker="o", markersize=5, color=COLORS[2], linewidth=2)
    ax.set(title="(c) 模块 B：独立诊断响应误差", xlabel="服务器优化步数", ylabel=r"响应 MSE（$\times 10^{-6}$）")
    ax.set_xticks(steps)
    ax.set_ylim(2.45, 3.22)
    ax.text(0.97, 0.91, "20 → 50 步：响应误差未继续下降", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#475569")

    ax = axes[1, 1]
    for mode, color in zip(MODES, COLORS):
        history = data["original_validation_history"][mode]
        ax.plot([r["epoch"] for r in history], [r["loss"] for r in history],
                color=color, linewidth=1.8, label=mode)
        best = min(history, key=lambda r: r["loss"])
        ax.scatter(best["epoch"], best["loss"], color=color, s=42, zorder=3, edgecolors="white")
    ax.set(title="(d) 模块 C：原 20 epochs 验证曲线", xlabel="目标域训练 epoch", ylabel="验证 BCE")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper right", ncol=2, fontsize=9.5)
    for ax in axes.flat:
        ax.grid()
        ax.set_axisbelow(True)
    fig.text(0.075, 0.035,
             "A：图、遮蔽和 EMA 目标随训练变化，不能据此证明饱和。B：重放固定每轮上传与探针，并非完整联邦轨迹对照。C：圆点为最低验证 BCE。",
             fontsize=10, color="#64748B")
    save_figure(fig, output, "budget")


def plot_validation(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.72, bottom=0.18, wspace=0.22)
    fig.suptitle("已完成预算配置：验证指标比较", fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.5, 0.89, "服务器均为 10 步  |  seed = 2026  |  同一固定 MolHIV 验证子集", ha="center", color="#64748B")
    specs = (("loss", "验证 BCE（越低越好）", (0.088, 0.094)),
             ("roc_auc", "验证 ROC-AUC（越高越好）", (0.51, 0.77)),
             ("average_precision", "验证 AP（越高越好）", (0.06, 0.20)))
    markers = ("o", "s", "D")
    handles = []
    for ax, (key, title, limits) in zip(axes, specs):
        for index, config in enumerate(data["completed_configs"]):
            ys = [i + (index - 1) * 0.21 for i in range(len(MODES))]
            xs = [config["modes"][mode]["validation"][key] for mode in MODES]
            handle = ax.scatter(xs, ys, c=COLORS[index], marker=markers[index],
                                s=67, edgecolors="white", linewidths=0.7,
                                label=config["label"], zorder=3)
            if key == "loss":
                handles.append(handle)
        ax.set(title=title, xlim=limits, ylim=(3.5, -0.5))
        ax.set_yticks(range(len(MODES)), labels=MODES)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        for y in range(len(MODES)):
            ax.axhline(y, color="#F1F5F9", zorder=0, linewidth=14)
    axes[0].set_xticks([0.088, 0.090, 0.092, 0.094])
    axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    axes[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    for ax, key, config_index, mode, offset, digits in (
        (axes[0], "loss", 2, "adapter", (9, -19), 6),
        (axes[1], "roc_auc", 2, "adapter", (-5, -24), 6),
        (axes[2], "average_precision", 1, "none", (-55, 18), 6),
    ):
        value = data["completed_configs"][config_index]["modes"][mode]["validation"][key]
        y = MODES.index(mode) + (config_index - 1) * 0.21
        ax.annotate(f"{value:.{digits}f}", (value, y), xytext=offset,
                    textcoords="offset points", color=COLORS[config_index], fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": COLORS[config_index]})
    fig.legend(handles, [c["label"] for c in data["completed_configs"]],
               loc="upper center", bbox_to_anchor=(0.5, 0.835), ncol=3, columnspacing=2.7)
    fig.text(0.07, 0.065, "每个点均取最低验证 BCE 选中的同一个 epoch；ROC-AUC/AP 未另选最佳值。最多 50 epochs 的配置使用 patience=10。",
             color="#64748B", fontsize=10)
    fig.text(0.07, 0.025, "这些是单种子的验证结果；20 轮 / 20 步尚未完成，未纳入比较。", color="#64748B", fontsize=10)
    save_figure(fig, output, "validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Reuse the exported figure data JSON")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "pic")
    args = parser.parse_args()
    data = read_json(args.data) if args.data else collect_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{PREFIX}-data.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    configure_style()
    plot_budget(data, args.output_dir)
    plot_validation(data, args.output_dir)


if __name__ == "__main__":
    main()
