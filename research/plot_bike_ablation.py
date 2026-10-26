"""Plot completed BIKE-Inflow ablations without loading models or changing results."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments/h1/ablation/summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/h1_ablation")
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    if report["status"] != "completed_bike_validation":
        raise ValueError("Only completed validation results may be plotted.")

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 11})

    arms = ("full", "global_long", "no_utility")
    labels = ("完整方法", "全局均值\n长期记忆", "无预测效用\n辅助监督")
    colors = ("#cf5148", "#8493aa", "#51a59b")
    aggregates = {item["arm"]: item for item in report["aggregate"]}
    # Check the exact source plotted: three adapter seeds and sample SD (ddof=1).
    for arm in arms:
        rows = [item for item in report["per_seed"] if item["arm"] == arm]
        if sorted(item["seed"] for item in rows) != [2021, 2022, 2023]:
            raise ValueError(f"Incomplete or duplicated seed set: {arm}")
        for metric in ("mae", "rmse", "mse"):
            values = [item[metric] for item in rows]
            actual = aggregates[arm][metric]
            if abs(statistics.mean(values) - actual["mean"]) > 1e-9:
                raise ValueError(f"Mean mismatch: {arm}/{metric}")
            if abs(statistics.stdev(values) - actual["sample_std"]) > 1e-9:
                raise ValueError(f"Sample SD mismatch: {arm}/{metric}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.3))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.77, bottom=0.30, wspace=0.30)
    fig.suptitle("BIKE-Inflow：两个创新点的消融实验", fontsize=19, y=0.965)
    fig.text(0.5, 0.885, "3 个种子均值 ± 样本标准差｜误差棒不是置信区间｜指标越低越好", ha="center", color="#555555", fontsize=11)

    for ax, metric in zip(axes, ("mae", "rmse", "mse")):
        means = [aggregates[arm][metric]["mean"] for arm in arms]
        stds = [aggregates[arm][metric]["sample_std"] for arm in arms]
        ceiling = max(mean + std for mean, std in zip(means, stds))
        ax.bar(range(3), means, yerr=stds, color=colors, width=0.62,
               capsize=5, error_kw={"ecolor": "#30343b", "elinewidth": 1.3}, zorder=3)
        for idx, (mean, std) in enumerate(zip(means, stds)):
            ax.text(idx, mean + std + ceiling * 0.035, f"{mean:.3f}\n± {std:.3f}",
                    ha="center", va="bottom", fontsize=10, linespacing=1.45)
        ax.set_title(metric.upper() + " ↓", fontsize=14, pad=14)
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, ceiling * 1.25)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e7ec", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a0a7b2")
        ax.tick_params(axis="both", length=0, pad=7)

    fig.text(0.055, 0.165, "对照① 全局均值长期记忆：仅将早于最近 48 小时的历史替换为全站均值，近期自身历史不变。", fontsize=10, color="#444444")
    fig.text(0.055, 0.115, "对照② 无预测效用辅助监督：仅移除检索辅助损失，保留可学习检索与预测损失。", fontsize=10, color="#444444")
    fig.text(0.055, 0.055, "评估：全测试集 48 小时预测；同一冻结基线下的 3 个改进分支种子；每组 5 轮（预算截止，未确认收敛）。", fontsize=9, color="#666666")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output_dir / f"ablation_bars.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
