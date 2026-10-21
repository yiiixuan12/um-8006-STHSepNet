# LICENSES

This repository bundles **modified** code from the official **STH-SepNet** project and a
small amount of third-party code. It also references external datasets and model weights
that are **not redistributed here**. Everything below records source, licence, and the
exact files affected. When in doubt, keep the original licences and notices in place.

## 1. Upstream STH-SepNet (modified) — MIT

- Source: https://github.com/jiawenchen10/STHSepNet
- Upstream commit (frozen reference): `e1b4a02ca10a4cc4a56c4ad4c105445f33db6e9a`
- Licence: MIT (see `LICENSE` at repository root / copied to `LICENSES-STH-SepNet-MIT.txt`).
  MIT permits use, copy, modification, and redistribution with attribution.
- Files: all code under `src/` that originates from the upstream repository
  (`run_main.py`, `models/`, `layers/`, `utils/`, `data_provider/`, `baselines/`,
  `ds_config_zero2.json`), **modified** for this course project as documented in
  `configs/` and `report/` (dtype/float32 fix, `layers` import fix, seed handling,
  checkpoint loading fix, optional DeepSpeed, and the added KAN prediction head).
- Modification diff / provenance: see `report/` (section on deviations) and git history
  of this repository.

## 2. pykan / KANLayer (unmodified third-party) — MIT

- Source: PyTorch implementation of Kolmogorov-Arnold Networks
  (https://github.com/KindXiaoming/pykan).
- Files: `src/kan/KANLayer.py`, `src/kan/spline.py` (B-spline basis functions). These are
  kept byte-for-byte as upstream (only `src/kan/utils.py` was trimmed to the needed
  `sparse_mask` helper, and `src/kan/KANLinear.py` is our thin `nn.Linear`-compatible
  wrapper around `KANLayer`).
- Licence: MIT (attribution preserved in file headers).

## 3. Datasets (referenced, not redistributed) — see `data/README.md`

- **PEMS03 / METR-LA / BJ / PV / WT**: original source links and licences are recorded in
  `data/README.md`. These are traffic/energy public benchmarks with their own licences;
  we do not redistribute them in this repo.
- **BIKE / inflow & outflow**: bundled inside the upstream STH-SepNet repository
  (`dataset/Bike/`). See `data/README.md` for provenance and checksums.

## 4. Pre-trained LLM weights (referenced, not redistributed) — see `data/README.md`

- **BERT** (`google-bert/bert-base-uncased`) and **GPT-2** (`gpt2`): downloaded from the
  Hugging Face Hub for this project. We do not commit weights; `data/README.md` gives the
  download command, version, and checksum procedure. They are subject to their own model
  card licences (Apache 2.0 for BERT; MIT for GPT-2).

## 5. This repository's own content

- All documentation and tests authored for this course project are provided under the
  same MIT licence as the upstream code, unless stated otherwise. Citation of the paper:

> Chen, Jiawen; Shao, Qi; Chen, Duxin; Yu, Wwenwu. *Decoupling Spatio-Temporal Prediction:
> When Lightweight Large Models Meet Adaptive Hypergraphs.* KDD 2025.

---
**Notice.** Third-party components retain their original copyright and licence texts.
No licence statement here supersedes the original licences.
