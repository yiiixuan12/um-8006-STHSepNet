# SECURITY

This repository is a course project (UM CISC8006). It handles public research data and
publicly released pre-trained model weights only. No personal data, no credentials, and
no restricted material are stored or expected here.

## Secrets & credentials

- **Never commit** real credentials, API keys, tokens, `.env`, private keys, or sensitive
  local files. Real values belong in your local `.env` (git-ignored); commit only
  `.env.example` with placeholders.
- Before every assessed release, run a secret scan over the whole tree (see
  `scripts/scan_secrets.sh`). Example on Linux:
  ```bash
  grep -rInE '(api[_-]?key|secret|token|password|BEGIN (RSA|OPENSSH) PRIVATE KEY)' \
    --exclude-dir=.git --exclude=.env.example . || echo "no obvious secrets"
  ```
- If a secret is ever exposed: revoke/rotate it immediately, remove it from history, and
  record the incident in the amendment log.

## Data egress & external APIs

- The default pipeline is fully offline once the datasets and weights are downloaded.
- Only optional tools (e.g. wandb logging) make external calls. Keep track of which data
  is sent where; the agent ledger (`AI_USAGE.md`) records what was sent to each external
  API/agent.
- No scraping, human-subject data collection, or biometric/sensitive personal data is
  part of this project. If the scope ever changes to include such data, prior instructor
  and applicable UM approval is required.

## Model & dataset provenance

- Datasets and model weights are kept **outside Git** and retrieved via authorized
  instructions (see `data/README.md`). Their source, version, licence and checksum are
  recorded there.
- Restricted assets are referenced and retrieved, never committed.

## Reporting a vulnerability / issue

This is a teaching repository; there is no production service. For concerns about the
underlying upstream project, please report to the official STH-SepNet repository. For
course-submission concerns, contact the course instructor directly.

Last reviewed: 2026-10-21 (Week 1, pre-freeze).
