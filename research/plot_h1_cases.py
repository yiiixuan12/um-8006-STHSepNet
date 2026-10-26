"""Redraw the published original-unit 24-hour cases without a GPU or weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "experiments/h1/cases/paired_24h_predictions.npz")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/h1/cases/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/h1_cases")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    selection = report["case_selection"]
    with np.load(args.predictions, allow_pickle=False) as data:
        gt = data["ground_truth"]
        baseline = data["baseline_raw"]
        ours = data["improved_raw"]
        times = data["timestamps_ns"]
        seeds = data["seeds"]
        origins = data["origins"]
    if baseline.shape != ours.shape or baseline.shape[:3] != (3, 3, 24):
        raise ValueError("Expected three paired seeds, three cases, 24 hours")
    if gt.shape != baseline.shape[1:] or not np.array_equal(seeds, [2021, 2022, 2023]):
        raise ValueError("Ground-truth or seed dimensions differ from the published cohort")
    if not np.array_equal(origins, selection["origins"]):
        raise ValueError("Case selection differs from the published predeclared origins")
    if not all(np.isfinite(x).all() for x in (gt, baseline, ours)):
        raise ValueError("Nonfinite published predictions")
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.arange(24)
    for case in range(3):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        date = np.datetime_as_string(np.datetime64(int(times[case, 0]), "ns"), unit="D")
        fig.suptitle(f"BIKE-Inflow: {date} | three independently paired seeds", fontsize=14)
        for ax, node in zip(axes, selection["nodes"]):
            b, o = baseline[:, case, :, node], ours[:, case, :, node]
            ax.plot(x, gt[case, :, node], color="black", linewidth=1.8, label="Ground truth")
            for values, color, name in ((b, "#3974b7", "Corrected baseline"), (o, "#d34b43", "H1")):
                ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.14)
                ax.plot(x, values.mean(0), color=color, linewidth=1.8, label=f"{name}: 3-seed mean")
            ax.set_title(f"Station {node}", loc="left", fontsize=10)
            ax.set_ylabel("Hourly inflow\n(original units)")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=8, ncol=3)
        axes[-1].set_xticks(range(0, 24, 2))
        axes[-1].set_xlabel("Forecast hour (single origin; no future observations used)")
        fig.text(0.5, 0.015, "Shading: actual min/max across 3 seeds, not a confidence interval. No clipping or smoothing.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        for suffix in ("png", "pdf"):
            output = args.output / f"paired_24h_case_{case + 1}.{suffix}"
            fig.savefig(output, dpi=180)
            print(output)
        plt.close(fig)


if __name__ == "__main__":
    main()
