# H1 results — yixuan wang

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

## Recompute metrics without data or GPU

The package includes 26 original per-window error arrays: five fixed baseline tests, 15 main adapter tests, and six additional ablation tests. Each file contains only `origin`, `window_mae` and `window_mse`, not original input series. The full-method ablation is the same BIKE-Inflow main result and is not duplicated again. With NumPy installed, run from the repository root:

```powershell
python research/check_h1_results.py --root results/h1
```

This independently recomputes test MAE, MSE and RMSE from the published window errors and requires exact equality with all 26 JSON report records. It also checks finite arrays, origin/window counts, consistent main/ablation test windows, and exits nonzero on failure. No BERT, dataset, CUDA or model checkpoints are needed. This checks arithmetic consistency of published evaluation evidence, not a fresh model inference or proof of training correctness.
