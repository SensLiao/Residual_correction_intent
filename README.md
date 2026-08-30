<div align="right"><a href="README.zh-CN.md">简体中文</a></div>

<p align="center"><img src="docs/hero.png" alt="Residual Correction Intent banner" width="100%"></p>

<p align="center"><strong>From a sparse scribble to an object-referenced, grammar-legal correction program for interactive whole-body PET/CT lesion segmentation.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20code-a78bfa?style=flat-square" alt="Research code">
  <img src="https://img.shields.io/badge/PyTorch-2.6-a78bfa?style=flat-square" alt="PyTorch 2.6">
  <img src="https://img.shields.io/badge/nnU--Net-v2-a78bfa?style=flat-square" alt="nnU-Net v2">
  <img src="https://img.shields.io/badge/Python-3.10-a78bfa?style=flat-square" alt="Python 3.10">
  <img src="https://img.shields.io/badge/tests-772-a78bfa?style=flat-square" alt="772 tests">
  <img src="https://img.shields.io/badge/license-all%20rights%20reserved-a78bfa?style=flat-square" alt="All rights reserved">
</p>

## Overview

**Residual Correction Intent** is the research code for the Honours thesis *"From Sparse Corrections to Textual Intent: State-Relative PET/CT Residual Correction."* It studies **state-conditioned executable correction programs (SCEP)** for interactive whole-body PET/CT lesion segmentation. Author: **Ruixuan Liao** (Honours researcher, University of Sydney), also co-author of three 2026 peer-reviewed journal papers in medical imaging and feature selection.

The core idea: a sparse scribble says **where** a correction is wanted, but not **which** object to edit or **which** operation to run. The system compiles the scribble, the current mask and the state into an object-referenced, grammar-legal correction program — learned **state-relatively, without ground-truth access** — and that program then governs a constrained residual edit.

## ✨ Highlights

- **Intent, not just location** — a scribble marks where; the compiler resolves *which object* and *which operand* it refers to.
- **State-relative program compiler** — picks the object and operand but **never predicts the operation**; the operation's authority comes from the cue sign.
- **A small, legal grammar (v1)** — programs are drawn from `GROW_LOCAL` / `COMPLETE` / `CREATE_NEW` / `TRIM` / `DELETE` / `REPAIR`.
- **Constrained residual editing** — a 13-channel editor where `ADD` is a monotone union and `REMOVE` applies hard complement protection.
- **Reproducibility by construction** — three physically separate data lanes, an immutable prediction artifact, leakage-aware splits, a SHA-256 manifest, and commit-pinned dependencies (see below).
- **Extensively tested** — 772 test functions guarding the contracts above.

## 🏗 How it works

<p align="center"><img src="docs/method.png" alt="Residual Correction Intent method diagram" width="100%"></p>

The pipeline turns a cue into a bounded edit:

1. **Inputs** — PET/CT, the current mask, and a signed scribble.
2. **Program Compiler** (state-relative) — selects the object and operand and emits a program; it **never predicts the operation** (authority comes from the cue sign).
3. **Object-grounded legal program** — a grammar-v1 program: `GROW_LOCAL` / `COMPLETE` / `CREATE_NEW` / `TRIM` / `DELETE` / `REPAIR`.
4. **Constrained Residual Editor** (13-channel) — applies the edit under guardrails: `ADD` = monotone union, `REMOVE` = hard complement protection.
5. **Output** — the corrected mask.

## 🔬 Reproducibility & guardrails

- **Three physically separate data lanes** — inference-visible / label-only / evaluation-only.
- **Immutable prediction artifact** — written *before* evaluation, so predictions cannot be tuned to the metric.
- **Leakage-aware evaluation** — patient-clustered bootstrap and leakage-aware splits.
- **Integrity manifest** — a SHA-256 manifest over the tracked artifacts.
- **Pinned dependencies** — nnU-Net v2 and autoPET V pinned by commit.
- **Dataset** — PSMA-PET/CT, 597 cases / 378 patients, CC BY-NC 4.0.
- **Test suite** — 772 test functions.

## 🧰 Tech stack

| Area | Technologies |
| --- | --- |
| Language | Python 3.10 |
| Deep learning | PyTorch (torch 2.6), nnU-Net v2 |
| Scientific | NumPy, nibabel, SciPy |
| Data | PSMA-PET/CT (597 cases / 378 patients, CC BY-NC 4.0) |

## 🚀 Getting started

This repository is best read as **auditable research code**, not a one-command reproduction. The full training and evaluation environment — CUDA, a vendored nnU-Net tree, the autoPET V package, and the licensed dataset — is **not runnable off the original machine**. What you *can* run locally are the tests and lint that guard the contracts:

```bash
python -m pytest -q tests
python -m ruff check scripts tests
```

## 🧪 Testing

```bash
python -m pytest -q tests
```

The suite contains **772 test functions** that enforce the data-lane separation, grammar legality, and residual-edit guardrails described above.

## 📌 Project status

This is honest, in-progress research — the state of each piece is tracked deliberately:

- **v2 six-class baseline — FROZEN** (byte-identical).
- **v3 program compiler — audited implementation candidate.**
- **Data materialisation — awaiting an independent receipt.**
- **J0–J9 experiments — not started.**
- **Performance results — none.** The repository asserts **no** performance results.

A manuscript is in preparation.

## 📄 License

A LICENSE will accompany the manuscript release; until then the code is **all rights reserved** (no `LICENSE` file is present yet). A `CITATION.cff` will be added. The dataset (PSMA-PET/CT) is licensed **CC BY-NC 4.0** and is not redistributed here.

<p align="center"><sub>Built by <a href="https://github.com/SensLiao">Ruixuan "Sens" Liao</a> · USYD Advanced Computing (Honours)</sub></p>
