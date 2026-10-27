# Baseline tests

Run from the repository root in the training environment:

```powershell
python -m unittest tests.test_baseline tests.test_coursework tests.test_coursework_early_stop tests.test_training_scripts
```

These tests use synthetic CSVs, small CPU tensors and temporary outputs. No BERT download, original datasets or GPU training is needed. Installed runtime dependencies are required.

| Coverage | Test module |
|---|---|
| All five dataset node counts, hourly/five-minute intervals, author split rounding | `test_baseline` |
| Seconds/microseconds/nanoseconds converted to explicit nanoseconds | `test_baseline` |
| Training-only normalization, constant nodes, preserved gaps, invalid data/adjacency rejection | `test_baseline` |
| 48-step history/targets and chronological boundaries; future values excluded from inputs | `test_baseline`, `test_coursework` |
| Actual baseline evaluation: original units, zero targets, all nodes/horizons, incomplete final batch, global RMSE, saved window errors | `test_baseline` |
| Deterministic loader order, complete window coverage | `test_baseline` |
| CPU baseline runner → train → checkpoint save/reload → validation/test → report; no adapter construction; overwrite/source-resume guards; resumed dropout training matches uninterrupted parameters, optimizer and RNG | `test_baseline` |
| Five-epoch patience, best checkpoint restore, optimizer/RNG retention on resume | `test_coursework_early_stop` |
| Compact checkpoint parameters and buffers | `test_coursework` |
| Five-dataset/three-seed publication aggregation, sample SD and detection of altered error arrays | `test_baseline` |
| Seed queues, completed-run skipping, smoke rejection and PowerShell plan mode | `test_training_scripts` |

The CPU integration test replaces only the large model and GPU transfer/precision helpers with a small trainable model. It exercises the actual baseline training and evaluation code, but does not certify BERT loading. The synthetic metric fixture independently expects MAE=12.8, MSE=456 and RMSE=sqrt(456), with results invariant to batch size 2/4/5.

## Optional real BERT/GPU integration

Prepare the local datasets and `huggingface/google_bert/` first, then run:

```powershell
$env:RUN_BASELINE_GPU = '1'
python -m unittest tests.test_baseline_gpu
Remove-Item Env:RUN_BASELINE_GPU
```

This explicitly runs one-batch, one-epoch baseline training on BIKE-Outflow and PEMS03, checking checkpoint save/reload, finite metrics, 48-step test outputs, and trainable LoRA parameters across all 12 BERT layers while original BERT parameters remain frozen. Outputs are temporary and removed after the test. Smoke metrics are not accuracy evidence. Without the environment flag, this test is skipped.

To check the published baseline evidence without training:

```powershell
python research/export_baseline_results.py --check
```

That command verifies the package under `results/baseline_jinqi/`; it checks arithmetic consistency of saved evidence rather than running inference. Other test modules cover H1 and KAN separately. Some of those modules use pytest, installed by `scripts/setup_training.ps1`.
