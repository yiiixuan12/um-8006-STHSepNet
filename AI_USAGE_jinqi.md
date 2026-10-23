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
