# CONTRIBUTIONS

Team contributions and evidence, cross-checked against repository history,
experiments registry, and agent ledger. Update this file before the Week 12
freeze with the real member names and their commits.

## Team & roles (fill in)

| Member | Role / responsibilities | Evidence (commits, logs, experiments) |
|---|---|---|
| {Member A} | Reproduction lead; data pipeline; baseline MLP runs | commits; `results/logs/baseline_s*.txt` |
| {Member B} | KAN head implementation; dtype/`layers` fixes; tests | commits touching `src/kan/`, `src/models/ST_SepNet.py`, `tests/` |
| {Member C} | Improvement design + full evaluation; report/presentation | commits; `results/logs/kan_s*.txt`; `report/`, `presentation/` |
| {Member D} | Static/adaptive branch audit; experiment registry; AI_USAGE | `tests/test_static_adaptive.py`, `experiments/` |

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
