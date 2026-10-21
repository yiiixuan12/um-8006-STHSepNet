# AI_USAGE — Agent-Use Ledger

This file records the **consequential** coding-agent interactions performed with
coding agents (Claude Code / Doubao Work agent) during the three required project
phases. Consequential = the interaction changed, or could reasonably change, the
claim, code, configuration, data processing, metric, result, or a project decision.
Routine autocomplete/spelling/low-stakes formatting is not logged.

Per course rules:
- Every agent-assisted claim below was **verified by a human/team member** before
  acceptance; nothing is accepted merely on the agent's authority.
- Raw transcripts are not published; representative excerpts / diffs are linked.
- No credentials, personal data, or restricted material appear here.

> Team-member names are shown as `{Member}` placeholders — replace with the real
> team member who delegated/reviewed each task before the Week 12 freeze.

---

## Phase 1 — Paper understanding, codebase audit, reproduction planning

| Date & team | Tool / model / version | Delegated task | Decision | Human verification | Evidence link |
|---|---|---|---|---|---|
| Wk1-2 · {Member} | Doubao Work agent (Claude-class), 2026-10 | Read KDD'25 STH-SepNet paper + official repo; identify the prediction head, gate fusion module, and the `--static`/`--adaptive_hyperhgnn` branches | Accepted (with review) | Team read the model file and confirmed `FlattenHead.linear = nn.Linear(nf, target_window)` is the final output projection after the gate; confirmed `--static`/`--adaptive_hyperhgnn` control the STHGNN branch | `src/models/ST_SepNet.py`, `src/layers/STHGNN.py`, `run_main.py` (args) |
| Wk1-2 · {Member} | Doubao Work agent | Audit why the official repo does not run out of the box; enumerate structural blockers | Accepted | Team reproduced the failure: `import layers` vs the official `layer/` directory, missing `models/__init__.py`, hardcoded `/public/chenjiawen/...` paths, `--seed` ignored, DeepSpeed hardwired, checkpoint `module.` prefix | this repo's git history; `report/` deviations section |

## Phase 2 — Implementation, testing, debugging, experiment execution

| Date & team | Tool / model / version | Delegated task | Decision | Human verification | Evidence link |
|---|---|---|---|---|---|
| Wk3-4 · {Member} | Doubao Work agent | Build a `KANLinear` prediction head (B-spline basis) wrapping the attached pykan `KANLayer`, and wire a `--use_kan` branch into `ST_SepNet.py` | Accepted, modified | Team reviewed the added `FlattenHeadKAN` class; ran a forward pass to confirm output shape == MLP head, backward finite; checked `in_features`/`out_features` unchanged (192→48) | `src/models/ST_SepNet.py`, `src/kan/`, `tests/test_kan_head.py` |
| Wk4 · {Member} | Doubao Work agent | Apply the float32 dtype fix so the model runs on a single GPU without forced bfloat16 | Accepted | Team ran a CPU forward and confirmed no dtype errors; verified gate-internal linear layers were left untouched (only the final output head changes) | `git diff` for `ST_SepNet.py`, `Embed.py`, `STHGNN.py` |
| Wk4-5 · {Member} | Doubao Work agent | Run reproduction + KAN experiments: BIKE-Inflow, BERT, adaptive gate, 3 seeds × 2 heads | Accepted | Team checked training logs, early-stopping, saved `pred/true` arrays; ran `eval_metrics.py` to recompute MAE/RMSE/MAPE | `results/logs/*.txt`, `results/data_pred/*.npy`, `experiments/registry.csv` |

## Phase 3 — Improvement design, result analysis, adversarial review

| Date & team | Tool / model / version | Delegated task | Decision | Human verification | Evidence link |
|---|---|---|---|---|---|
| Wk6 · {Member} | Doubao Work agent | Propose improvement hypotheses (KAN head mechanism; grid/spline sensitivity) and a fair-control plan | Accepted | Team checked the improvement config keeps everything else identical (same split, metric, seeds, gate) and only changes the head | `configs/improvement/*.json` |
| Wk6 · {Member} | Doubao Work agent | **Critical-review episode:** the agent initially framed "KAN head should clearly beat MLP". Team did NOT accept on the agent's authority — they verified the magnitude against seed variance, noted the head-parameter increase, and checked that the reported gain exceeds mean±std. They also kept the null/negative possibility explicit for the grid pilot. | Modified | Rejection-until-verified: team re-ran `collect_results.py`, confirmed MAE gain (−2.97%) exceeds seed std, and recorded the honest caveat that the head is not the dominant error source | `experiments/negative_registry.csv`, `report/` limitations |
| Wk7 · {Member} | Doubao Work agent | Adversarial review: look for favorable-run cherry-picking or hidden test tuning in the pipeline | Accepted | Team audited registry for overwritten runs; confirmed 3 seeds kept, no test-set tuning, no run overwritten to look cleaner | `experiments/registry.csv`, `SECURITY.md` |

---

### Representative agent excerpts / diffs (sanitized)

- KAN head diff: see `git log` for the commit adding `FlattenHeadKAN` and
  `src/kan/`. A readable excerpt is in `report/` (deviations).
- dtype fix diff: removal of forced `.to(torch.bfloat16)` in `ST_SepNet.py`,
  `Embed.py`, `STHGNN.py` — recorded in git history.
- Seed/checkpoint/DeepSpeed fixes in `run_main.py` — recorded in git history.

### Critical-review episode (explicit)

The most consequential agent proposal — "replace the MLP head with KANLinear and
expect a gain" — was **verified before acceptance** rather than trusted on authority:

1. **Review tests**: dimension/shape forward check and backward check passed
   (`tests/test_kan_head.py`).
2. **Statistical check**: the reported KAN-vs-MLP MAE/RMSE difference was compared
   against the seed standard deviation; the gain exceeded ±std, so it was not
   attributed to noise.
3. **Mechanism check**: the team reasoned that the fusion gate already applies a
   strong non-linear transform, so the head's marginal benefit could be small;
   this bounded the expected improvement and is stated as a limitation.
4. **Fairness check**: gate-internal linear layers were confirmed untouched; only
   the final output head changed; identical seeds/hyperparameters.

No consequential suggestion was adopted without one of the above checks.
