# H1 — sanitized discussion and decision summary

Owner: **yixuan wang**. Compiled on **2026-10-26**.

This is a retrospective summary of consequential H1 discussions in the project
chat, not a verbatim transcript, a newly run experiment, or evidence of human
verification. Earlier messages without recoverable event dates are marked
undated; no model version, exact quotation or manual check is invented. OpenAI
Codex assisted with research planning, code, diagnostics, execution, analysis and
writing. Team members remain responsible for the submitted work. KAN belongs to
**jinfeng guo** and its historical ledger is outside this update's verification
scope. No credentials, personal machine paths or process identifiers are included.

## 1. Hypotheses and scope

**Question.** Could STH-SepNet be improved by letting its temporal branch retain
individual road patterns and use historical context more meaningfully, rather
than processing only the global-average sequence?

**Discussion.** Early, undated dialogue considered factor prompts, conditional
spatial/hypergraph memory, node-level temporal tokens, long histories and tests of
pretrained-versus-random backbones. The user repeatedly asked whether these ideas
actually improved prediction, whether they made the pretrained model useful, and
whether the system was becoming unnecessarily complex.

**Retained decision.** The final H1 concerns node-level multi-period memory and
prediction-utility-supervised historical retrieval. Neighbor-road conditioning
was not retained as a third claimed contribution. The final project does not
claim that BERT pretraining is advantageous, that language semantics explains the
gain, that physical adjacency proves causality, or that the model has human-like
spatial understanding. BERT is inherited machinery, not itself the innovation.

Evidence: [H1 scope](../../experiments/h1/README.md),
[fast model](../../models/periodic_bert_fast.py).

## 2. Diagnosis and the retained method

In undated earlier dialogue, weak neural-branch gains triggered discussion of
long-term information being bypassed, historical retrieval lacking a useful
learning signal, limited adaptation and inadequate spatial conditioning. These
were hypotheses to investigate, not all proven causes. The final pipeline retains
a numeric periodic reference as well as neural memory; it is not simply a BERT
tokenizer replacement.

Each target node retains recent, within-day and cross-day information. Historical
candidates contain context and their already-observed continuations. Retrieval
can weight candidates or abstain. Auxiliary utility supervision uses training
prediction error to guide selection; at inference it does not receive the current
future target. The forecast loss also trains the retrieval route. Consequently,
removing the auxiliary loss is not equivalent to removing historical retrieval.

Attempts to strengthen correction, align historical patterns or add neighboring
signals did not justify retaining every proposed contribution. The user ultimately
chose a smaller coursework study rather than indefinite model expansion. Neither
all complete-model gains nor the numeric reference's benefit can be assigned
exclusively to language pretraining.

Evidence: [periodic model](../../models/periodic_bert_v4.py),
[memory policy](../../models/periodic_memory_policy.py),
[training functions](../../research/run_coursework.py).

## 3. Final protocol and efficiency

Recorded schedule discussions on 2026-10-20–23 changed the practical budget with
the user's authorization. The final published main matrix covers PEMS03,
METR-LA, BJ500, BIKE-Inflow and BIKE-Outflow, with adapter seeds 2021/2022/2023.
Each city has one frozen baseline; three adapter seeds are **not** three complete
independent baseline-plus-adapter repetitions. Target-based train/validation/test
splits are 7:1:2, normalization uses training data, and validation raw MSE selects
best weights. Final adapters have a five-total-epoch cap and patience five. Their
cap stops do not establish convergence. Baseline budgets are recorded separately.

H1 sees more history than the 48-step baseline: 2064 steps for five-minute data
and 240 hourly steps for BIKE. BIKE's 48-step forecast means 48 hours, not four
hours. Its daily period is 24; lag-two-to-eight-day candidates have complete
continuations available before the current prediction time. No hourly-to-five-
minute resampling or true holiday/direction labels were added.

The approved fast model compresses BERT memory and uses coarse/fine candidate
selection; that **changes the architecture** and is not described as an exact
execution optimization. Later changes to fixed-index caching and statistics
transfer preserve the scientific computation. Recorded CUDA equality probes
checked outputs, loss, gradients, optimizer updates and random state; they are
distinct from an accuracy experiment. A probe fixture's shared optimizer-step
state required correction before that strict equivalence check passed. This was
not evidence that a formal training checkpoint had been corrupted. Original
results and failure evidence were retained locally.

Evidence: [portable protocol](../../research/run_h1.py),
[hourly interface](../../models/periodic_hourly.py),
[execution layer](../../research/fast_execution.py),
[published per-run records](../../experiments/h1/main/per_seed.csv).
Local monitoring and probe artifacts are not falsely linked as though this Git
package publishes the entire historical execution tree.

## 4. Main results and interpretation

The five-dataset, fifteen-adapter matrix completed before the additional BIKE
study. The published table replaces early exploratory H1 pilot numbers. For
example, PEMS03's fixed corrected baseline MAE/RMSE are 21.6194/35.8744, versus
18.2208 ± 0.1031 / 32.0173 ± 0.1739 for H1. These are local corrected results,
not official paper metrics.

METR-LA is an important adverse result: mean MAE improves from 10.3596 to 8.9335,
but mean RMSE increases from 17.8723 to 17.9998 and MSE from 319.4191 to 324.0029.
MAE weights absolute deviations linearly, whereas squared error emphasizes large
mistakes; the directions need not agree. This mathematical explanation does not
prove the specific reason for the model's tail errors. The user requested
analysis without altering the running model, and the final report must not
silently hide this trade-off, change thresholds or select only favorable seeds.

Evidence: [main results](../../experiments/h1/main/table.md),
[main JSON reports](../../experiments/h1/main/reports),
[metric recomputation](../../research/check_h1_results.py).

## 5. Raw-flow cases and bias diagnosis

The 2026-10-24 BIKE-Inflow study separately constructed three independently
paired baseline/full-method seeds. It is not the main study's shared-backbone
cohort. The model directly forecasts 48 hours and the case plots show its first
24 hours, without future observations being fed back into inference.

Dates were fixed from eligible early/middle/late test midnights and stations
245/252/191 from training-only mean quartiles. Original-unit values use training
node mean/standard deviation after the internal inverse transformation; there
is no extra clipping, smoothing or rounding. Plot bands are the actual three-
seed minimum/maximum, not confidence intervals. Cases were not replaced because
they looked unfavorable.

When high-flow stations looked overestimated and smooth, recorded checks examined
source values, station indices, normalization and time alignment. They did not
support a simple restoration or timestamp error as the main explanation. The
discussion identified inherited baseline bias, overly positive historical
additions, averaging and coarse periodic representations as plausible issues.
These explanations were not all separated by controlled retraining. The user
accepted discussing these limitations for coursework and asked only to finish
the authorized three-seed cases and ablations. Proposed fixes were **not**
silently implemented as additional successful experiments.

Evidence: [case report](../../experiments/h1/cases/report.json),
[selection](../../experiments/h1/cases/selection.json),
[raw-unit curves](../../experiments/h1/cases/paired_24h_curves.csv),
[second case, including unfavorable behavior](../../experiments/h1/cases/paired_24h_case_2.png).

## 6. Ablations and claim revision

The completed BIKE-Inflow ablation compares three arms over three adapter seeds
under one fixed baseline and the same five-epoch cap:

| Arm | Mean MAE | What is actually changed |
|---|---:|---|
| Full H1 | 4.4859 | Retain the completed method |
| Global-mean older history | 4.8124 | Replace only the older 192 of 240 hourly steps with all-station raw mean; preserve own recent 48 hours; refit numeric calibration |
| No auxiliary utility supervision | 4.5014 | Set auxiliary utility CE weight to zero; retain trainable retrieval, values, abstention and forecast gradients |

Node-specific older history wins on all three reported metrics in all three
seeds. This supports the value of individual long-term history in this setting,
not a separate proof for every token scale or a proof that BERT decoded periodic
semantics. Utility supervision has a small mean benefit, but seed 2023 reverses
the direction on all metrics. Thus the discussion explicitly rejected the
overclaim that both contributions are independently and significantly proven.
The ablation bars use sample standard deviation, not a statistical significance
test. Forecast curves remain supplementary case illustrations.

Evidence: [ablation table](../../experiments/h1/ablation/table.md),
[per-seed results](../../experiments/h1/ablation/per_seed.csv),
[bar chart](../../experiments/h1/ablation/ablation_bars.png),
[control implementation](../../research/bike_validation_support.py).

## 7. Release and submission

On 2026-10-26 the user authorized publishing H1 code, completed evidence and a
concise usage guide while preserving teammates' KAN work. H1 is attributed to
yixuan wang and KAN to jinfeng guo. The release separates H1's frozen baseline
operators, supplies a portable train/resume/evaluate interface, and publishes
five-dataset tables, reports, window-error evidence, ablation bars and raw-flow
cases. Raw datasets, pretrained BERT and model/optimizer checkpoints are excluded;
their existence must not be inferred from Git result tables.

The recorded publication checks were 13 targeted interface/information-control
checks, a real CUDA checkpoint/raw-flow smoke check, 78 metric recomputations and
one focused checkpoint/resume review. They are release checks performed with
agent assistance, not human certification, full-repository testing or new model
training. The evidence exporter sanitizes machine-local paths and process IDs
without altering scientific numerical values. The original experiment trees
were not overwritten.

The report update reconciles the final H1 pipeline, input budgets,
shared-backbone seeds, completed results, mixed utility evidence and case
limitations. It preserves the teammate's KAN result paragraphs and numerical
tables and treats KAN as an independent study, not a component ablation of H1.
All 13 rendered report pages were visually checked. No training was rerun for
this writing update. This summary and AI_USAGE.md disclose the assistance used
and the interpretations accepted or rejected; they do not replace the team's
review of the final coursework submission.

Evidence: [portable entry](../../research/run_h1.py),
[release tests](../../tests/test_h1_release.py),
[publication provenance](../../experiments/h1/provenance.json),
[result guide](../../experiments/h1/README.md),
[updated report](../../report/CISC8006_Final_Report_STH-SepNet.docx).

## 8. H1 script delivery update on 2026-10-27

The user identified the missing H1 script deliverables. Thin PowerShell wrappers
now cover dependency setup, interface smoke checks, the existing Python run entry
and published-result checks. They do not introduce another training implementation
or silently launch a full queue. Paths are repository-relative and native failures
stop subsequent commands. PyTorch selection remains explicit rather than inferred
from a user's machine. Seven wrapper checks and the default wrapper's nine H1
interface checks passed; no dependencies were installed, no model was trained and
no GPU experiment was run. See the [script guide](../../scripts/README.md).
