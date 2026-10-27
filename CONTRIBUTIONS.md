# CONTRIBUTIONS

Team contributions and evidence, cross-checked against repository history,
experiments registry, and agent ledger. Update this file before the Week 12
freeze with the real member names and their commits.

## Team & roles (fill in)

| Member | Role / responsibilities | Evidence (commits, logs, experiments) |
|---|---|---|
| Jinqi Liu | Five-dataset, three-seed baseline reproduction; dataset and training-entry fixes; installation, smoke and seed-run scripts; baseline tests and result publication | `research/run_coursework.py`, `scripts/training_runs.py`, `tests/test_baseline.py`, `tests/test_baseline_gpu.py`, `results/baseline_jinqi/`, `logs/agents/2026-10-27-training-scripts.md`, `logs/agents/2026-10-27-baseline-tests.md` |
| jinfeng guo | H3: KAN prediction head and independent BIKE-Inflow evaluation | Existing `src/kan/`, `src/models/ST_SepNet.py`, KAN tests and reports |
| yixuan wang | H1: node-level multi-period memory, utility-supervised retrieval, five-dataset evaluation, BIKE ablations and 24-hour cases | `research/run_h1.py`, `models/periodic_bert_fast.py`, `models/sthsepnet_h1.py`, `layers/h1/`, `experiments/h1/` |


## Contribution evidence checklist

- [ ] Every member has at least one identifiable commit touching code or
      documentation they are responsible for.
- [ ] Reproduction runs (3 seeds) logged by the reproduction lead.
- [ ] KAN experiments logged by the improvement member.
- [ ] Registry rows link each run to the responsible member's git commit.
- [ ] Agent ledger (`AI_USAGE.md`) attributes each consequential interaction to a
      real member who delegated/reviewed it.
- [ ] Peer feedback and defense evidence confirm the roles (instructor triangulates;
      peer ratings are not a substitute).

> Note: contribution marks require **sustained, identifiable** contribution
> supported by commits, roles, and logs. Do not leave `{Member}` placeholders in
> the final submission.
