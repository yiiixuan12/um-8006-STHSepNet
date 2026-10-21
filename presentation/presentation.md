# Presentation — STH-SepNet Reproduction & KAN Head Improvement

Team presentation (Wk13-15 Symposium) and individual defense. Target format per course:
15 min talk (claim → reproduction → agent workflow → improvement → limitations →
demonstration) + 10 min questions/individual defense + 2 min transition. All members speak.

## Deck outline
1. **Title + claim** — one testable claim; reproduction tier; frozen protocol.
2. **Paper & method** — decoupled temporal/spatial, gate fusion, MLP head.
3. **Reproduction** — protocol, 3 seeds, MAE/RMSE vs paper, discrepancy diagnosis.
4. **Official-code audit** — layers fix, dtype, seed, checkpoint; static vs adaptive branch.
5. **Agent workflow** — AI_USAGE.md: 3 phases + critical-review episode.
6. **Improvement** — KAN head (mechanism), pilots (grid sweep), full eval vs baseline,
   parameter efficiency, robustness.
7. **Limitations** — MAPE caveat, head not dominant error source, no grid adaptation.
8. **Demonstration** — bounded smoke test / config-log walkthrough (not a costly live run).

## Demo instructions
- `bash scripts/smoke_test.sh` (offline CPU) — verified runnable by peer auditor.
- `python scripts/collect_results.py` — regenerates the comparison table from saved arrays.
- Optional: `python scripts/eval_metrics.py baseline_s2021 kan_s2021`.

## Files
- `presentation/deck.pptx` (final deck, added before freeze)
- `presentation/demo_notes.md` (recorded fallback + walkthrough)
