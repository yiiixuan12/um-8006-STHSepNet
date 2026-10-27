"""Recompute 26 published H1 test records using only NumPy and window errors.

This verifies published arithmetic evidence, not model inference or training.
No dataset, BERT, Torch, CUDA, model weights or source experiment tree is needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


DATASETS = ("pems03", "metr_la", "bj500", "bike_inflow", "bike_outflow")
SEEDS = (2021, 2022, 2023)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_record(npz_path, report_path, stage, reference_origins=None):
    expected = read_json(report_path)["stages"][stage]["test"]
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data.files) != {"origin", "window_mae", "window_mse"}:
            raise AssertionError(f"Unexpected arrays in {npz_path}")
        origin = data["origin"]
        absolute = data["window_mae"]
        squared = data["window_mse"]
        if any(x.ndim != 1 for x in (origin, absolute, squared)):
            raise AssertionError(f"Non-vector errors in {npz_path}")
        if not (len(origin) == len(absolute) == len(squared) == expected["windows"]) or not len(origin):
            raise AssertionError(f"Window count mismatch in {npz_path}")
        if not np.issubdtype(origin.dtype, np.integer) or not np.all(np.diff(origin) > 0):
            raise AssertionError(f"Invalid or unordered origins in {npz_path}")
        if not all(np.isfinite(x).all() for x in (absolute, squared)) or np.any(absolute < 0) or np.any(squared < 0):
            raise AssertionError(f"Invalid error values in {npz_path}")
        if reference_origins is not None and not np.array_equal(origin, reference_origins):
            raise AssertionError(f"Test windows differ in {npz_path}")
        metrics = {"mae": float(np.mean(absolute)), "mse": float(np.mean(squared))}
        metrics["rmse"] = math.sqrt(metrics["mse"])
        for key, value in metrics.items():
            if value != expected[key]:
                raise AssertionError(f"Exact metric mismatch: {npz_path}, {key}: computed={value!r}, report={expected[key]!r}")
        return origin.copy(), {"artifact": npz_path.as_posix(), "windows": len(origin), **metrics}


def check(root):
    records = []
    origins = {}
    for dataset in DATASETS:
        folder = root / "main/errors" / dataset
        origin, record = check_record(folder / "baseline_test.npz", root / "main/baseline_reports" / f"{dataset}.json", "baseline")
        origins[dataset] = origin
        records.append(record)
        for seed in SEEDS:
            _, record = check_record(folder / f"seed_{seed}_test.npz", root / "main/reports" / dataset / f"seed_{seed}.json", "improved", origin)
            records.append(record)
    for arm in ("global_long", "no_utility"):
        for seed in SEEDS:
            _, record = check_record(root / "ablation/errors" / arm / f"seed_{seed}_test.npz",
                                     root / "ablation/reports" / arm / f"seed_{seed}.json", "improved", origins["bike_inflow"])
            records.append(record)
    if len(records) != 26:
        raise AssertionError("Expected exactly 26 test records")
    print(json.dumps({"status": "passed", "records_checked": len(records),
                      "metric_comparisons": len(records) * 3, "equality": "exact",
                      "needs_gpu_or_original_data": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/h1"))
    args = parser.parse_args()
    check(args.root)
