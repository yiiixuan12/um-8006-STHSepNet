# AI_USAGE — Consequential-Interaction Ledger

This file records disclosed **consequential** coding-agent assistance for the
STH-SepNet coursework. A consequence is an interaction that changed,
or could reasonably change, a claim, the code, a config, data processing, a metric,
a result, or a project decision. Routine autocomplete / spelling / low-stakes
formatting is not logged.

**Responsibility and disclosure:**
- The team is responsible for reviewing and accepting the submission. An agent-run
  code/result check is not human verification; the H1 entries below explicitly
  distinguish recorded decisions, artifact checks and claims not established.
- The pre-existing AI-KAN-001 through AI-KAN-007 entries, including their historical
  human-verification statements, are retained as supplied. This H1 update does not
  independently re-audit those KAN statements or their historical evidence paths.
- Raw transcripts are not published. H1 entries link a sanitized retrospective
  discussion summary under `logs/agents/` and concrete repository artifacts. The
  summary is not a verbatim transcript or an exhaustive event-by-event record.
- No credentials, personal data, or restricted material appear in this file or its
  logs.


All evidence paths are relative to this repository root.

---

## Ledger

### AI-KAN-001 — 2026-08-22 — GUO JINFENG
- **Stage:** Paper understanding / reproduction planning
- **Tool / model / version:** DeepSeek V4.1 Flash · 2026-08-22 · default settings
  (temperature not exposed → *not exposed*)
- **Delegated task:** Interpret the KDD'25 paper *Decoupling Spatio-Temporal
  Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs*
  (STH-SepNet); identify improvement points; assess whether KAN can improve it.
- **Summary (sanitized):** the agent laid out the decoupled architecture (lightweight
  LLM for temporal, adaptive hypergraph for spatial, gated fusion), and flagged three
  diagnosis-driven improvements: (i) representation stage — `AvgPool(X)` before the
  LLM obscures per-segment intra-day / cross-day periodic patterns; (ii) fusion stage —
  the GCN–hypergraph coefficient `r` is a global scalar that cannot adapt per node/time;
  (iii) prediction stage — `FlattenHead.linear = nn.Linear(192, 48)` uses fixed
  activations. The agent judged `KANLinear` feasible while keeping `in=192, out=48`
  unchanged, and additionally suggested *enabling `update_grid` and adding history input*.
- **Decision: Modified.** Accepted the reading and the three improvement points, and
  accepted the "replace only the prediction head, keep 192→48" feasibility. **Rejected**
  the `update_grid` + extra-history suggestion: the former would add a second adaptive
  variable (breaking the single-controlled-variable rule for H3), the latter would break
  the information-matched comparison.
- **Human verification:** compared paper Eqs. (7)(22)(26)–(27) against the official code
  for `AvgPool`, the fusion coefficient `r`, and the `FlattenHead` call path; confirmed
  MLP head 9,264 params, KAN head 103,680 (Δ 94,416); confirmed H3 input 48 steps /
  output 48 steps, identical to baseline.
- **Evidence:**
  - Code: `src/models/ST_SepNet.py` (`FlattenHead`, gate call), `src/layers/STHGNN.py` (fusion coefficient `r`)
  - Report: `report/final_report.md` 
  - Agent log: `logs/agents/2026-08-22_kan_paper_review.md`
- **Sanitization:** none required; only public paper and repo code.

### AI-KAN-002 — 2026-08-28 — GUO JINFENG
- **Stage:** Codebase audit / reproduction planning
- **Tool / model / version:** Gemini 3 Pro · 2026-08-28 · default settings (*not exposed*)
- **Delegated task:** audit the insertion point of `FlattenHead`, `fusion_gate=adaptive`,
  and the `--use_kan` switch; check for hidden dimension change, parameter sharing, or
  checkpoint-key conflicts.
- **Summary (sanitized):** agent gave the call chain fused-output → `FlattenHead` →
  prediction, suggested a conditional branch inside `FlattenHead` switching
  `nn.Linear(192,48)` vs `KANLinear(192,48)` while leaving the gate-internal linear
  layers untouched, and reminded that checkpoint-key prefix and dtype fixes must be
  shared with the baseline.
- **Decision: Accepted.** Keeps the only controlled variable = head type (information-matched
  design); environment fixes are shared between baseline and KAN, not scientific variables.
- **Human verification:** checked the call chain in `src/models/ST_SepNet.py` and configs;
  ran a sanity forward+backward on a random batch `(2,48,295)` for both heads; checked
  param counts; ran a 1-epoch smoke train → checkpoint save/load → cleanup, exit 0.
- **Evidence:**
  - Code: `src/models/ST_SepNet.py` (`--use_kan` branch), `src/kan/KANLinear.py`
  - Sanity: `tests/synthetic_sanity.py`
  - Agent log: `logs/agents/2026-08-28_kan_code_audit.md`
- **Sanitization:** none; stack traces had local absolute paths replaced with `[REDACTED: local path]`.

### AI-KAN-003 — 2026-09-05 — GUO JINFENG
- **Stage:** Implementation / testing / debugging
- **Tool / model / version:** DeepSeek V4.1 Flash · 2026-09-05 · default settings (*not exposed*)
- **Delegated task:** implement the `KANLinear(192, 48; grid_size=5, spline_order=3)`
  wrapper, the `--use_kan` switch and `update_grid` config; debug dtype/device errors in KAN backward.
- **Summary (sanitized):** agent provided a `KANLinear` wrapper draft and again
  suggested enabling `update_grid` with dynamic grid refinement during training.
- **Decision: Modified.** Accepted the wrapper and switch; **rejected `update_grid`** —
  the frozen H3 protocol requires `update_grid=False`, otherwise grid adaptation becomes
  a second scientific variable.
- **Human verification:** reviewed the diff; confirmed `update_grid=False`,
  `grid_size=5`, `spline_order=3` written to `configs/improvement/kan_adaptive_bike_bert.json`;
  ran sanity; checked KAN head 103,680 params and total +0.09%; no gradient anomalies.
- **Evidence:**
  - Code: `src/kan/KANLinear.py`, `src/models/ST_SepNet.py` (`--use_kan`)
  - Config: `configs/improvement/kan_adaptive_bike_bert.json` (grid=5, spline=3, update_grid=False)
  - Tests: `tests/test_kan_head.py`, `tests/synthetic_sanity.py`
  - Agent log: `logs/agents/2026-09-05_kan_impl_debug.md`
- **Sanitization:** none; stack traces had local absolute paths removed.

### AI-KAN-004 — 2026-09-18 — GUO JINFENG
- **Stage:** Experiment execution / debugging
- **Tool / model / version:** Gemini 3 Pro · 2026-09-18 · default settings (*not exposed*)
- **Delegated task:** help write the 3-seed (`2021/42/7`) MLP/KAN comparison runs; check
  seed propagation, Windows spawn guard, checkpoint-key prefix, and dtype adaptation.
- **Summary (sanitized):** agent recommended using the same training script with only
  `--use_kan` appended, a separate output dir per seed, and no overwriting of existing
  runs; and recomputing MAE/RMSE/robust MAPE from the saved `.npy` with
  `scripts/eval_metrics.py` / `collect_results.py` instead of trusting printed log values.
- **Decision: Accepted.** Consistent with the protocol (same split / BERT / gate / budget,
  only variable `--use_kan`); environment fixes shared between baseline and KAN.
- **Human verification:** checked run commands; confirmed seed propagated into model init
  and DataLoader; confirmed per-seed logs, pred/true arrays, and `report.json` saved
  independently; metrics recomputed from `.npy`.
- **Evidence:**
  - Entry: `src/run_main.py` (`--seed`, `--use_kan`)
  - Recompute: `scripts/eval_metrics.py`, `scripts/collect_results.py`
  - Registry: `experiments/registry.csv` (runs R-KAN-2021 / R-KAN-42 / R-KAN-7)
  - Arrays: `results/data_pred/*.npy`; summary `results/comparison_table.csv`, `results/comparison_summary.json`
  - Agent log: `logs/agents/2026-09-18_kan_run_debug.md`
- **Sanitization:** none.

### AI-KAN-005 — 2026-10-08 — GUO JINFENG
- **Stage:** Result analysis / adversarial review
- **Tool / model / version:** DeepSeek V4.1 Flash · 2026-10-08 · default settings (*not exposed*)
- **Delegated task:** analyze the BIKE-Inflow 3-seed results; judge whether KAN is
  "fully better" than the MLP baseline, and explain the anomalous robust-MAPE direction.
- **Summary (sanitized):** the agent initially proposed reporting "KAN fully better than
  MLP"; after cross-checking data it acknowledged KAN's robust MAPE = 80.175 ± 0.191 is
  *higher* than MLP's 77.417 ± 0.827, which does not support "fully better".
- **Decision: Modified / Rejected overclaim.** Rejected "fully better"; reframed as a
  **metric-dependent mixed result** (supported on MAE/RMSE, not on robust MAPE). Because
  41.3% of BIKE samples have |y|<1 after de-normalization, naive MAPE is not reportable;
  robust MAPE (|y|≥1) is kept and the +3.6 pts degradation is reported honestly.
- **Human verification:** recomputed MAE/RMSE/robust MAPE from saved `.npy`; checked the
  robust-MAPE denominator |y|≥1; compared seed-by-seed — KAN 3/3 seeds beat all 3 baseline
  seeds on MAE/RMSE but the robust-MAPE direction is opposite.
- **Evidence:**
  - Recompute: `scripts/eval_metrics.py`, `scripts/collect_results.py`
  - Data: `results/data_pred/*.npy`
  - Report: `report/final_report.md` 
  - Agent log: `logs/agents/2026-10-08_kan_result_review.md`
- **Sanitization:** none.

### AI-KAN-006 — 2026-10-15 — GUO JINFENG — **critical-review episode**
- **Stage:** Improvement design / adversarial review
- **Tool / model / version:** Gemini 3 Pro · 2026-10-15 · default settings (*not exposed*)
- **Delegated task:** adversarially review H3 — find confounders, decide whether to
  enable `update_grid` or raise KAN capacity to amplify the gain, and whether H3 should
  compete with H1 as the main improvement.
- **Summary (sanitized):** the agent suggested enabling `update_grid`, increasing
  `grid_size`, and positioning KAN as the "main improvement" to compete with H1.
- **Decision: Rejected**, on human-verified grounds (not the agent's say-so):
  1. `update_grid` would add a second adaptive variable, violating H3's single-controlled-variable rule;
  2. no compute-matched control ⇒ cannot attribute any gain;
  3. H1's PEMS03 pilot relative MAE gain (−16.38%) vs H3's (−2.97%) ⇒ by the frozen
     selection rule H1 goes to full evaluation, H3 stays as a controlled ablation;
  4. adding capacity would change the parameter-count comparison and weaken the
     information-matched interpretation.
  Final state kept: `update_grid=False`, `grid_size=5`, `spline_order=3`, H3 as a
  BIKE-Inflow controlled ablation only.
- **Human verification:** read the `KANLinear` implementation and
  `configs/improvement/kan_adaptive_bike_bert.json`; checked H3-vs-MLP input/output dims,
  param counts, seeds, and training budget; cross-checked H1/H3 pilot gains against the
  selection rule; confirmed no run was overwritten.
- **Evidence:**
  - Config: `configs/improvement/kan_adaptive_bike_bert.json`, `configs/improvement/kan_grid_ablation_pilot.json`
  - Registry: `experiments/registry.csv`
  - Report: `report/final_report.md` 
  - Agent log: `logs/agents/2026-10-15_kan_adversarial_review.md`
- **Sanitization:** none.

### AI-KAN-007 — 2026-10-20 — GUO JINFENG
- **Stage:** Improvement design / ablation planning
- **Tool / model / version:** DeepSeek V4.1 Flash · 2026-10-20 · default settings (*not exposed*)
- **Delegated task:** design a fair ablation for H3 that separates the "head-type effect"
  from the "parameter/capacity effect", and plan what goes into the report's limitations.
- **Summary (sanitized):** the agent proposed at least reporting MLP-head vs KAN-head,
  KAN param count, total-parameter share, per-seed paired comparison, and robust MAPE;
  and, if compute allowed, adding a "KAN capacity matched to MLP parameters" control.
- **Decision: Modified.** Accepted MLP-vs-KAN, per-seed, param-count, and robust MAPE.
  **Rejected** claiming the "capacity-matched KAN" as a completed experiment, since it
  was not run under the current budget; it is listed as a limitation / future work.
- **Human verification:** checked params 9,264 vs 103,680 (total Δ 94,416, +0.09%);
  checked 3/3-seed direction; confirmed the robust-MAPE degradation is written into limitations.
- **Evidence:**
  - Report: `report/final_report.md` 
  - Registry: `experiments/registry.csv`
  - Agent log: `logs/agents/2026-10-20_kan_ablation_plan.md`
- **Sanitization:** none.

## H1 ledger — yixuan wang

Tool for the entries below: **OpenAI Codex**. Exact historical model identifiers and
sampling settings are not asserted where they are not recoverable. Entries
AI-H1-001 through AI-H1-007 were summarized on 2026-10-26 from the recorded dialogue
and published artifacts; subsequent entries carry their own dates. Dates
refer to recorded discussion/execution periods, not invented dates for earlier work.

### AI-H1-001 — Undated earlier dialogue — hypothesis selection

- **Task:** investigate how a pretrained temporal backbone could use node-specific
  long-term and spatial information instead of only a global average.
- **Decision:** narrow H1 to node-level multi-period memory and utility-supervised
  historical retrieval. Drop the proposed neighboring-road third contribution from
  the final study; do not claim that BERT pretraining or human-like spatial
  understanding has been demonstrated.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#1-hypotheses-and-scope),
  [model](models/periodic_bert_fast.py), [published scope](experiments/h1/README.md).
- **Verification boundary:** design rationale is not evidence that each component
  improves accuracy. Earlier exploratory routes are not presented as completed
  successful innovations or assigned reconstructed dates.

### AI-H1-002 — Undated earlier dialogue — diagnose the long-term route

- **Task:** analyze weak gains, short-term bypass, ineffective historical selection
  and unstable pretrained-versus-random comparisons.
- **Decision:** retain a numeric periodic reference route; let the neural route
  refine forecasts using current and historical context. Use prediction-value
  supervision for historical selection and retain the option to abstain. Do not
  attribute the complete method's gain solely to BERT or token design.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#2-diagnosis-and-the-retained-method),
  [periodic model](models/periodic_bert_v4.py),
  [memory policy](models/periodic_memory_policy.py),
  [training functions](research/run_coursework.py).
- **Verification boundary:** these are consequential implementation/design
  choices; the final ablations, not the motivation alone, determine independent
  support. No stable language-pretraining advantage is claimed.

### AI-H1-003 — 2026-10-20–23 — practical protocol and runtime decisions

- **Task:** complete feasible multi-dataset experiments under the assignment
  deadline without silently discarding seeds, nodes or data.
- **Decision:** final main scope is five datasets, three adapter seeds per dataset,
  one shared frozen baseline per city and a five-total-epoch adapter cap. Select
  best weights by validation raw MSE. Keep longer-input and hourly-BIKE differences
  explicit. Distinguish an authorized fast **architecture change** from later
  **equivalent execution optimizations**.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#3-final-protocol-and-efficiency),
  [portable entry](research/run_h1.py),
  [hourly model](models/periodic_hourly.py),
  [execution layer](research/fast_execution.py),
  [per-seed records](experiments/h1/main/per_seed.csv).
- **Verification boundary:** equal-output execution checks do not prove that the
  earlier fast architecture is mathematically identical to the slow model.
  Five-epoch caps are budget stops, not confirmed convergence; three adapter seeds
  do not mean three independent end-to-end baseline trainings.

### AI-H1-004 — 2026-10-23–24 — main results and adverse findings

- **Task:** aggregate the completed five-dataset matrix and explain why lower MAE
  need not imply lower squared error.
- **Decision:** preserve all results, including METR-LA's improved mean MAE but
  worse RMSE/MSE. Report corrected local baselines rather than call them the
  original paper's numbers. Do not select favorable seeds or change the frozen
  model in response to test-set outcomes.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#4-main-results-and-interpretation),
  [main table](experiments/h1/main/table.md),
  [main reports](experiments/h1/main/reports),
  [metric checker](research/check_h1_results.py).
- **Verification boundary:** published window-error arithmetic can be independently
  recomputed; it is not new model inference or proof that BERT caused the gain.
  Higher squared error is consistent with larger tail mistakes but does not by
  itself identify their causal mechanism.

### AI-H1-005 — 2026-10-24 — independent paired 24-hour cases

- **Task:** show restored hourly flow against ground truth using three baseline
  seeds and three paired full-method seeds, and investigate overprediction/smoothing.
- **Decision:** use predeclared dates/stations and the first 24 hours of a direct
  48-hour forecast. Preserve raw-unit negative/overestimated outputs without
  cosmetic clipping or smoothing. Keep the independent paired-case cohort separate
  from shared-backbone ablations; do not start proposed repair experiments.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#5-raw-flow-cases-and-bias-diagnosis),
  [case restoration report](experiments/h1/cases/report.json),
  [original-unit values](experiments/h1/cases/paired_24h_curves.csv),
  [case figure](experiments/h1/cases/paired_24h_case_2.png).
- **Verification boundary:** recorded diagnostics did not support an inverse-scaling
  or time-alignment error as the main explanation for the displayed gaps. Incorrect
  retrieval additions, coarse periodic representation and inherited baseline bias
  remain plausible explanations, not experimentally isolated causal findings.
  A few cases do not substitute for full-test evaluation.

### AI-H1-006 — 2026-10-24–26 — ablations and rejected overclaim

- **Task:** test the two proposed mechanisms with three seeds and summarize them
  using a bar chart rather than only forecast curves.
- **Decision:** compare full H1 against global-mean older history and removal of
  auxiliary utility supervision, keeping retrieval itself in the latter. Report
  mean MAE 4.4859 / 4.8124 / 4.5014, respectively. Reject the statement that both
  innovations are independently and significantly established: utility supervision
  has only a small mean gain and seed 2023 reverses direction on all three metrics.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#6-ablations-and-claim-revision),
  [ablation table](experiments/h1/ablation/table.md),
  [per-seed results](experiments/h1/ablation/per_seed.csv),
  [bar chart](experiments/h1/ablation/ablation_bars.png),
  [control implementation](research/bike_validation_support.py).
- **Verification boundary:** older node-specific history has consistent support in
  this BIKE-Inflow comparison. The control does not separately isolate all token
  scales, numeric calibration and neural encoding. Error bars are sample standard
  deviations, not confidence intervals or significance tests.

### AI-H1-007 — 2026-10-26 — publication and report reconciliation

- **Task:** publish only H1 code/results, add portable instructions and reconcile
  the report and this disclosure with the final evidence; preserve the other
  contributors' work. H1 owner is **yixuan wang**; KAN owner is **jinfeng guo**.
- **Decision:** isolate H1's frozen baseline/operators and publish compact results,
  original-unit case values and figures. Do not publish datasets, BERT or large
  checkpoints as though they were included. Replace early H1 pilot statements
  with the completed main/ablation evidence and disclose the longer input budget.
- **Evidence:** [discussion summary](logs/agents/h1_discussion_summary.md#7-release-and-submission),
  [portable entry](research/run_h1.py),
  [release tests](tests/test_h1_release.py),
  [publication provenance](experiments/h1/provenance.json),
  [published result guide](experiments/h1/README.md),
  [updated team report](report/CISC8006_Final_Report_STH-SepNet.docx).
- **Verification boundary:** the recorded release checks comprise 13 targeted
  checks, a real CUDA loading/raw-flow smoke check, 78 metric recomputations and
  one focused checkpoint/resume review. They are agent-executed release checks,
  not human approval, new accuracy training or full-repository certification.
  This disclosure update does not re-audit KAN. Machine-local paths/process IDs
  were sanitized without changing scientific values; original experiments remain
  unchanged. The team must review the final submission.
  The report update was rendered and all 13 pages were visually checked; the
  existing H3 result paragraphs and both numeric tables were checked for content
  preservation. These document checks did not rerun training or certify the
  teammate's scientific claims.


### AI-H1-008 — 2026-10-27 — portable PowerShell scripts

- **Task:** fill the missing H1 installation, smoke-check and reproduction script
  deliverables without changing teammates' entry points or scientific results.
- **Decision:** add thin PowerShell wrappers for dependency installation,
  stdlib-only smoke checks, the existing baseline/improved/ablation/evaluation
  interface, and published-result recomputation. Use repo-relative file paths,
  argument arrays, explicit native-exit checks and an execution-free `-PlanOnly`.
  Keep the old local coursework queue as a clearly labeled historical script;
  do not silently launch a full multi-city queue or guess a PyTorch/CUDA build.
- **Evidence:** [script guide](scripts/README.md),
  [shared execution helper](scripts/h1_common.ps1),
  [wrapper checks](tests/test_h1_scripts.py).
- **Verification boundary:** seven targeted wrapper checks passed under Windows
  PowerShell 5.1; the actual default smoke wrapper also ran the nine existing
  H1 interface tests successfully. Native argument forwarding and failure handling
  were checked without model construction. Installation was previewed, not run;
  no new accuracy training, CUDA validation, full-suite run or human approval is
  asserted. Scientific Python, checkpoints and published metrics were unchanged.

# AI_USAGE — Jinqi Consequential-Interaction Ledger

Records consequential agent assistance for Jinqi's dataset adaptation, experiment tracking, debugging, and result summaries. This ledger is separate from the KAN ledger in `AI_USAGE.md`.

Reconstructed on 2026-10-23 from the conversation history; the earlier ledger and logs were unavailable in the inspected repository versions. Dates identify the recorded work episodes, not recovered transcript timestamps. Historical model versions and sampling settings were not retained and are marked unknown. Evidence paths are relative to the repository root.

## Ledger

### AI-JINQI-001 — 2026-10-21 — Jinqi
- **Stage:** Implementation / testing / debugging / experiment execution support
- **Tool / model / version:** OpenAI Codex; historical model/version unknown; sampling settings not exposed.
- **Delegated task:** Preserve the baseline/improved command interface, adapt available datasets, automatically register experiments, and fix data-loading problems.
- **Summary (sanitized):** Extended dataset configuration for PEMS03, METR-LA, BJ500, BIKE-Inflow and BIKE-Outflow; added end-of-invocation experiment registration and provenance snapshots; normalized timestamp conversion to nanoseconds. Prepared a BJ500/METR-LA baseline queue for seeds 2021–2023. Kept BIKE-specific training settings and documented the hourly interpretation of the memory interval.
- **Decision: Modified.** Accepted dataset adaptation and tracking. At the user's request, removed started records: only completed, failed or caught interrupted outcomes are registered. Kept weights in the requested output directory. Smoke results are process checks, not accuracy evidence; no historical experiments are fabricated.
- **Human verification:** The user confirmed on 2026-10-23 that code and experiments had been manually reviewed and experiments passed. This records that confirmation; it does not assert that the subsequently merged code was revalidated.
- **Evidence:**
  - Code: [training entry](research/run_coursework.py), [registry](research/experiment_registry.py), [dataset adapter](research/coursework_data.py), [timestamp handling](research/traffic_data.py)
  - Tests: [registry tests](tests/test_experiment_registry.py), [coursework tests](tests/test_coursework.py)
  - Agent log: [2026-10-21-registry-datasets](logs/agents/2026-10-21-registry-datasets.md)
- **Sanitization:** Local absolute paths omitted; no credentials or raw transcripts included.

### AI-JINQI-002 — 2026-10-23 — Jinqi
- **Stage:** Implementation / testing / result aggregation
- **Tool / model / version:** OpenAI Codex; historical model/version unknown; sampling settings not exposed.
- **Delegated task:** Extend the coursework summarizer to all five supported dataset tasks and update the agent log and AI usage record.
- **Summary (sanitized):** Expanded default dataset coverage; added dataset/stage selection, including baseline-only summaries; retained per-seed records and mean/sample-standard-deviation summaries; checked missing runs and excluded smoke or mismatched reports from accuracy aggregation.
- **Decision: Accepted.** Missing runs remain missing rather than becoming zero-valued measurements; a single run does not establish across-seed variability. Aggregation does not launch training.
- **Human verification:** The user confirmed that manual code review and review of test results passed. The historical session recorded 11 relevant tests passing and a baseline aggregation of eight available runs with seven missing combinations. These are historical checks, not a fresh post-merge test run.
- **Evidence:**
  - Code: [summarizer](research/summarize_coursework.py)
  - Tests: [summarizer tests](tests/test_summarize_coursework.py), [coursework tests](tests/test_coursework.py)
  - Agent log: [2026-10-23-summary-all-datasets](logs/agents/2026-10-23-summary-all-datasets.md)
- **Sanitization:** Local absolute paths omitted; no credentials or raw transcripts included.

This ledger documents the available Jinqi episodes only. It does not claim that these two entries alone cover every course-required phase or constitute a completed adversarial-review episode.