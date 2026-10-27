"""Publish completed H1 evidence without training or changing source artifacts.

Only Python's standard library is required. Source paths inside public JSON are
replaced by portable ``source://`` references, not download links. Machine-local
process identifiers and file modification times are removed. Scientific numeric
values, including all training logs, are checked against the source before export.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from pathlib import Path


DATASETS = ("pems03", "metr_la", "bj500", "bike_inflow", "bike_outflow")
SEEDS = (2021, 2022, 2023)
METRICS = ("mae", "rmse", "mse")
ARMS = ("full", "global_long", "no_utility")
DISPLAY = {"pems03": "PEMS03", "metr_la": "METR-LA", "bj500": "BJ500",
           "bike_inflow": "BIKE-Inflow", "bike_outflow": "BIKE-Outflow"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def private_key(key):
    return key in {"pid", "process_id", "parent_process_id", "child_process_id"} or key.endswith("_pid") or "mtime" in key


def sanitize(value, source_root):
    if isinstance(value, dict):
        return {k: sanitize(v, source_root) for k, v in value.items() if not private_key(k)}
    if isinstance(value, list):
        return [sanitize(v, source_root) for v in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        root = str(source_root).replace("\\", "/").rstrip("/")
        normalized = re.sub(re.escape(root) + r"(?=/|$)", "source://", normalized, flags=re.I)
        normalized = normalized.replace("source:///", "source://")
        if re.search(r"(?i)\b[a-z]:/", normalized):
            return "external-local-path://" + normalized.rsplit("/", 1)[-1]
        return normalized
    return value


def numeric_values(value, prefix=()):
    if isinstance(value, dict):
        return {p: v for k, item in value.items() if not private_key(k)
                for p, v in numeric_values(item, prefix + (k,)).items()}
    if isinstance(value, list):
        return {p: v for i, item in enumerate(value)
                for p, v in numeric_values(item, prefix + (i,)).items()}
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite scientific numeric value: {prefix}")
        return {prefix: value}
    return {}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def export_json(src, dst, source_root):
    original = read_json(src)
    public = sanitize(original, source_root)
    if numeric_values(original) != numeric_values(public):
        raise AssertionError(f"Numeric content changed while sanitizing {src}")
    write_json(dst, public)
    return original


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_aggregate(rows, aggregate, key):
    for item in aggregate:
        selected = [r for r in rows if r[key] == item[key]]
        if len(selected) != 3 or {r["seed"] for r in selected} != set(SEEDS):
            raise AssertionError(f"Expected three distinct seeds: {item[key]}")
        for metric in METRICS:
            values = [r[metric] for r in selected]
            for expected, actual in ((statistics.mean(values), item[metric]["mean"]),
                                     (statistics.stdev(values), item[metric]["sample_std"])):
                if not math.isclose(expected, actual, rel_tol=1e-13, abs_tol=1e-13):
                    raise AssertionError(f"Aggregate mismatch: {item[key]} {metric}")


def verify_stage(report, row, stage="improved"):
    result = report["stages"][stage]
    if not result.get("status", "").startswith("completed"):
        raise AssertionError(f"Stage is incomplete: {row}")
    for metric in METRICS:
        if result["test"][metric] != row[metric]:
            raise AssertionError(f"Report/summary mismatch: {metric}")
    if stage == "improved" and any(result[k] != row[k] for k in ("epochs", "best_epoch", "stop_reason")):
        raise AssertionError("Training budget or best epoch mismatch")


def source_file(root, reference):
    normalized = reference.replace("\\", "/")
    marker = "/results_coursework/"
    if marker not in normalized:
        raise ValueError(f"Unknown source result reference: {reference}")
    return root / "results_coursework" / normalized.split(marker, 1)[1]


def main_table(summary):
    baseline = {r["dataset"]: r for r in summary["fixed_baseline_references"]}
    lines = ["# H1: full test-set results", "", "Mean ± sample standard deviation across three adapter seeds. Lower is better.", "",
             "| Dataset | Method | MAE | RMSE | MSE |", "|---|---|---:|---:|---:|"]
    for row in summary["aggregate"]:
        base = baseline[row["dataset"]]
        lines.append(f"| {DISPLAY[row['dataset']]} | Fixed corrected baseline (seed 2021) | " + " | ".join(f"{base[m]:.4f}" for m in METRICS) + " |")
        lines.append(f"| {DISPLAY[row['dataset']]} | H1 (three adapter seeds) | " + " | ".join(f"{row[m]['mean']:.4f} ± {row[m]['sample_std']:.4f}" for m in METRICS) + " |")
    lines += ["", "Each dataset uses one shared frozen baseline. The three seeds apply to adapter initialization and training order, not independent end-to-end repetitions.",
              "H1 reads additional history (2064 steps for five-minute data; 240 hourly steps for BIKE); baseline input is 48 steps. Therefore gains are not a matched-information test of the representation alone.",
              "All adapter runs reached the five-epoch cap; convergence was not confirmed. Validation raw MSE selects the best checkpoint. These are local corrected-baseline results, not the paper's official numbers.",
              "METR-LA improves mean MAE but worsens mean RMSE/MSE. Negative outcomes are retained.", ""]
    return "\n".join(lines)


def ablation_table(summary):
    names = {"full": "Full H1", "global_long": "Global-mean older history", "no_utility": "No auxiliary utility supervision"}
    lines = ["# BIKE-Inflow ablations", "", "Whole test-set 48-hour forecasts; mean ± sample standard deviation over three adapter seeds. Lower is better.", "",
             "| Arm | MAE | RMSE | MSE |", "|---|---:|---:|---:|"]
    for row in summary["aggregate"]:
        lines.append(f"| {names[row['arm']]} | " + " | ".join(f"{row[m]['mean']:.4f} ± {row[m]['sample_std']:.4f}" for m in METRICS) + " |")
    lines += ["", "![Ablation bars](ablation_bars.png)", "", "Error bars are sample standard deviations, not confidence intervals. All arms share one frozen baseline and have a five-epoch adapter cap.",
              "`global_long` replaces only the older 192 hours of the 240-hour history with the all-station raw mean; each target's own recent 48 hours are retained. This affects neural memory, candidate/teacher histories and the separately refitted numeric route. It is not a 'no history' or 'no retrieval' control.",
              "`no_utility` removes only the auxiliary utility cross-entropy. Trainable retrieval, candidate values, abstention and forecast-error gradients remain. It is not a 'no retrieval' control.",
              "Node-specific older history improves all metrics in all three seeds. Utility supervision gives a small average gain, but seed 2023 reverses the direction on all metrics: its independent benefit is not stable. This study does not establish language-pretraining superiority.", ""]
    return "\n".join(lines)


def copy_window_errors(source_root, output):
    """Copy 26 small error summaries, never original input series or weights."""
    matrix = source_root / "results_coursework/fast_matrix_v1"
    bike = source_root / "results_coursework/bike_validation_v1"
    copied = []
    for dataset in DATASETS:
        pairs = [(matrix / dataset / "seed_2021/baseline_test.npz",
                  output / "main/errors" / dataset / "baseline_test.npz")]
        pairs += [(matrix / dataset / f"seed_{seed}/improved_test.npz",
                   output / "main/errors" / dataset / f"seed_{seed}_test.npz") for seed in SEEDS]
        for src, dst in pairs:
            if not src.is_file() or src.stat().st_size == 0:
                raise AssertionError(f"Missing window-error artifact: {src.name}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(dst.relative_to(output).as_posix())
    for arm in ("global_long", "no_utility"):
        for seed in SEEDS:
            src = bike / "ablations" / arm / f"seed_{seed}/improved_test.npz"
            dst = output / "ablation/errors" / arm / f"seed_{seed}_test.npz"
            if not src.is_file() or src.stat().st_size == 0:
                raise AssertionError(f"Missing ablation error artifact: {src.name}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(dst.relative_to(output).as_posix())
    return copied


def document_window_errors(output, copied):
    readme = output / "README.md"
    content = readme.read_text(encoding="utf-8")
    marker = "## Recompute metrics without data or GPU"
    if marker not in content:
        content += """
## Recompute metrics without data or GPU

The package includes 26 original per-window error arrays: five fixed baseline tests, 15 main adapter tests, and six additional ablation tests. Each file contains only `origin`, `window_mae` and `window_mse`, not original input series. The full-method ablation is the same BIKE-Inflow main result and is not duplicated again. With NumPy installed, run from the repository root:

```powershell
python research/check_h1_results.py --root results/h1
```

This independently recomputes test MAE, MSE and RMSE from the published window errors and requires exact equality with all 26 JSON report records. It also checks finite arrays, origin/window counts, consistent main/ablation test windows, and exits nonzero on failure. No BERT, dataset, CUDA or model checkpoints are needed. This checks arithmetic consistency of published evaluation evidence, not a fresh model inference or proof of training correctness.
"""
        readme.write_text(content, encoding="utf-8")
    path = output / "provenance.json"
    provenance = read_json(path)
    provenance["window_error_evidence"] = {
        "records": 26, "fields": ["origin", "window_mae", "window_mse"],
        "source_window_errors_modified": False,
        "recompute_command": "python research/check_h1_results.py --root results/h1",
        "no_raw_input_series_or_model_checkpoints": True}
    provenance["copied_original_artifacts"] = sorted(set(provenance["copied_original_artifacts"] + copied))
    write_json(path, provenance)


def publish(source_root, output):
    matrix = source_root / "results_coursework/fast_matrix_v1"
    bike = source_root / "results_coursework/bike_validation_v1"
    main = read_json(matrix / "summary.json")
    ablation = read_json(bike / "summary.json")
    if main["status"] != "completed_15_adapter_runs" or main["missing"] or len(main["per_seed"]) != 15:
        raise AssertionError("The formal matrix is not complete")
    if ablation["status"] != "completed_bike_validation" or ablation["missing_ablations"] or len(ablation["per_seed"]) != 9:
        raise AssertionError("The ablations are not complete")
    if {r["dataset"] for r in main["per_seed"]} != set(DATASETS):
        raise AssertionError("Unexpected dataset matrix")
    if {r["arm"] for r in ablation["per_seed"]} != set(ARMS):
        raise AssertionError("Unexpected ablation matrix")
    verify_aggregate(main["per_seed"], main["aggregate"], "dataset")
    verify_aggregate(ablation["per_seed"], ablation["aggregate"], "arm")
    output.mkdir(parents=True, exist_ok=True)
    export_json(matrix / "summary.json", output / "main/summary.json", source_root)
    export_json(bike / "summary.json", output / "ablation/summary.json", source_root)
    write_csv(output / "main/per_seed.csv", main["per_seed"])
    baseline_rows = []
    for row in main["per_seed"]:
        src = matrix / row["dataset"] / f"seed_{row['seed']}" / "report.json"
        report = export_json(src, output / "main/reports" / row["dataset"] / f"seed_{row['seed']}.json", source_root)
        verify_stage(report, row)
    for ref in main["fixed_baseline_references"]:
        report = export_json(source_file(source_root, ref["source_report"]), output / "main/baseline_reports" / f"{ref['dataset']}.json", source_root)
        verify_stage(report, ref, stage="baseline")
        stage = report["stages"]["baseline"]
        baseline_rows.append({"dataset": ref["dataset"], "seed": ref["seed"], "shared_frozen_baseline": True,
                              "epochs": stage["epochs"], "best_epoch": stage["best_epoch"], "stop_reason": stage["stop_reason"],
                              **{m: ref[m] for m in METRICS}})
    write_csv(output / "main/baselines.csv", baseline_rows)
    public_ablation_rows = []
    for row in ablation["per_seed"]:
        report = export_json(source_file(source_root, row["source_report"]), output / "ablation/reports" / row["arm"] / f"seed_{row['seed']}.json", source_root)
        verify_stage(report, row)
        public_ablation_rows.append(sanitize(row, source_root))
    write_csv(output / "ablation/per_seed.csv", public_ablation_rows)
    case_report = export_json(bike / "cases/report.json", output / "cases/report.json", source_root)
    if case_report["status"] != "completed_three_paired_seeds" or case_report["seeds"] != list(SEEDS):
        raise AssertionError("Paired case cohort incomplete")
    export_json(bike / "cases/selection.json", output / "cases/selection.json", source_root)
    case_test_rows = []
    for pair in ablation["paired_full_test"]:
        report = export_json(source_file(source_root, pair["source_report"]), output / "cases/reports" / f"seed_{pair['seed']}.json", source_root)
        for method, stage in (("baseline", "baseline"), ("improved", "improved")):
            row = {"seed": pair["seed"], "method": method, **{m: pair[method][m] for m in METRICS}}
            verify_stage(report, row, stage="baseline") if method == "baseline" else None
            if any(report["stages"][stage]["test"][m] != row[m] for m in METRICS):
                raise AssertionError("Paired case report metric mismatch")
            row.update({k: report["stages"][stage][k] for k in ("epochs", "best_epoch", "stop_reason")})
            case_test_rows.append(row)
    write_csv(output / "cases/paired_full_test.csv", case_test_rows)
    copied = []
    for name in ("ablation_bars.png", "ablation_bars.pdf"):
        src, dst = bike / "figures" / name, output / "ablation" / name
        shutil.copyfile(src, dst)
        copied.append(str(dst.relative_to(output)).replace("\\", "/"))
    for name in ([f"paired_24h_case_{i}.{ext}" for i in (1, 2, 3) for ext in ("png", "pdf")]
                 + ["paired_24h_curves.csv", "paired_24h_predictions.npz"]):
        src, dst = bike / "cases" / name, output / "cases" / name
        if not src.is_file() or src.stat().st_size == 0:
            raise AssertionError(f"Missing case artifact: {src.name}")
        shutil.copyfile(src, dst)
        copied.append(str(dst.relative_to(output)).replace("\\", "/"))
    (output / "main/table.md").write_text(main_table(main), encoding="utf-8")
    (output / "ablation/table.md").write_text(ablation_table(ablation), encoding="utf-8")
    readme = """# H1 results — yixuan wang

This package publishes completed coursework evidence for **node-level multi-period memory and utility-supervised historical retrieval**. It does not contain KAN results or claim language-pretraining superiority.

## Results

- [Five-dataset main table](main/table.md), [15 adapter seeds](main/per_seed.csv), [five single frozen baselines](main/baselines.csv), and complete per-run JSON reports.
- [BIKE-Inflow ablation table and bar chart](ablation/table.md): full method, global-mean older history, and no auxiliary utility supervision, each with three adapter seeds.
- [24-hour case 1](cases/paired_24h_case_1.png), [case 2](cases/paired_24h_case_2.png), [case 3](cases/paired_24h_case_3.png), [pointwise original-unit values](cases/paired_24h_curves.csv), and [case selection / restoration report](cases/report.json).

## Interpretation and limits

The main matrix and ablations use one fixed seed-2021 baseline per dataset, with three **adapter** seeds. They are not three independent end-to-end runs. Every adapter reaches the five-epoch budget cap; convergence is not confirmed. Baseline and improved methods have different history lengths. Results are local corrected reproductions, not original-paper official numbers.

Node-specific older memory has consistent BIKE-Inflow ablation support. Additional utility supervision has a small, seed-dependent gain; seed 2023 is worse on every reported metric. METR-LA improves mean MAE but worsens RMSE/MSE. Do not claim every metric improves or both innovations are independently and significantly established.

The 24-hour case cohort is separate: three independently trained paired baseline/full-method seeds. A single direct 48-hour prediction is made, then its first 24 hours are displayed, without using observations from the forecast period. The three dates are selected from eligible test time boundaries, and stations 245/252/191 from training-only mean quartiles. Selection is not based on favorable predictions. Values are restored by `prediction * training_node_std + training_node_mean`; internal RevIN is already reversed. There is no extra clipping, smoothing or rounding. Shaded bands show the actual three-seed min/max range, not confidence intervals. The curves retain overprediction and smoothing limitations rather than hiding them.

## Re-export (no training)

From the repository root, run:

```powershell
python research/export_h1_results.py --source-root PATH_TO_COMPLETED_PROJECT --output results/h1
```

The source project must contain the completed `results_coursework/fast_matrix_v1` and `results_coursework/bike_validation_v1` trees. The exporter reads them without modification, verifies completed counts, report metrics and sample-standard-deviation aggregates, and copies already-generated figures and case values.

Published JSON removes machine-local process IDs and file modification times, and replaces absolute source paths with `source://...` provenance references. These references describe the original artifact location; they are not download URLs or assertions that the original tree is in this repository. All scientific numeric values and training logs are preserved. See [provenance.json](provenance.json). Raw datasets, BERT, best/last/optimizer checkpoints, local execution supervisors and caches are deliberately excluded from this evidence package. Availability of trained weights must not be assumed from this package.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    provenance = {"owner": "yixuan wang", "status": "published_completed_h1_evidence",
                  "source_cohorts": ["results_coursework/fast_matrix_v1", "results_coursework/bike_validation_v1"],
                  "source_artifacts_modified": False, "retraining_performed": False,
                  "main_adapter_reports": 15, "main_fixed_baselines": 5, "ablation_reports": 9, "independent_paired_cases": 3,
                  "sanitization": {"removed": "machine-local process IDs and file modification times only",
                                   "paths": "source:// relative provenance; external-local-path:// basename for outside paths",
                                   "scientific_numeric_values_and_training_logs_unchanged": True},
                  "verification": {"summary_status_complete": True, "report_metrics_equal_source_summary": True,
                                   "mean_and_sample_std_recomputed": True, "all_scientific_numeric_values_finite_and_preserved": True},
                  "copied_original_artifacts": copied,
                  "excluded_large_artifacts": ["raw datasets", "pretrained BERT", "model and optimizer checkpoints", "training caches"]}
    write_json(output / "provenance.json", provenance)
    error_artifacts = copy_window_errors(source_root, output)
    document_window_errors(output, error_artifacts)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    for p in files:
        if p.suffix in (".json", ".md", ".csv") and re.search(r"(?i)\b[a-z]:[\\/]", p.read_text(encoding="utf-8")):
            raise AssertionError(f"Machine absolute path leaked: {p.name}")
    print(json.dumps({"status": "exported", "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                      "main_adapter_reports": 15, "ablation_reports": 9, "paired_seeds": 3}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/h1"))
    parser.add_argument("--errors-only", action="store_true", help="Add window errors/docs to an already exported package without repeating JSON export checks")
    args = parser.parse_args()
    if args.errors_only:
        error_artifacts = copy_window_errors(args.source_root.resolve(), args.output.resolve())
        document_window_errors(args.output.resolve(), error_artifacts)
        print(json.dumps({"status": "window_errors_added", "records": len(error_artifacts)}))
    else:
        publish(args.source_root.resolve(), args.output.resolve())
