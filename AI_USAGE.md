# AI_USAGE — Consequential-Interaction Ledger

This file records every **consequential** coding-agent interaction performed for the
STH-SepNet improvement. A consequence is an interaction that changed,
or could reasonably change, a claim, the code, a config, data processing, a metric,
a result, or a project decision. Routine autocomplete / spelling / low-stakes
formatting is not logged.

**Course rules (binding):**
- Every agent-assisted claim below was **verified by a human/team member** before
  acceptance; nothing is accepted on the agent's authority.
- Raw transcripts are not published; each entry links a sanitized agent log under
  `logs/agents/` plus the concrete code/config/result artifacts that back the claim.
- No credentials, personal data, or restricted material appear in this file or its
  logs.


All evidence paths are relative to this repository root (`sths-repro/`).

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
  - Report: `report/final_report.md` §6.1 / §7
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
  - Report: `report/final_report.md` §5 / §8.2 / §10 (MAPE caveat)
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
  - Report: `report/final_report.md` §7 / §8 / §9 (selection rule and H3 controlled ablation)
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
  - Report: `report/final_report.md` §7 / §8.2 / §10 (params, per-seed, robust MAPE, limitations)
  - Registry: `experiments/registry.csv`
  - Agent log: `logs/agents/2026-10-20_kan_ablation_plan.md`
- **Sanitization:** none.

---

## Cross-cutting summary

- **Three required phases covered:** paper understanding/reproduction planning
  (AI-KAN-001, 002), implementation/testing/execution (AI-KAN-003, 004),
  improvement design/result analysis/adversarial review (AI-KAN-005, 006, 007).
- **Critical-review episodes (≥1 required):** AI-KAN-005 and AI-KAN-006 are explicit
  adversarial reviews. In AI-KAN-006 the agent's proposal ("enable `update_grid`, push
  KAN as main improvement") was **rejected on verified grounds**, and the over-claim
  "KAN fully better" (AI-KAN-005) was reframed as a metric-dependent mixed result.
- **Uniform guardrail across all entries:** `update_grid=False` and no extra history
  input were held constant; only the prediction head changes for H3, keeping the
  comparison information-matched.
- All seven agent logs are under `logs/agents/`; the decisions and human verifications
  above are the authoritative record and were not delegated to any agent.
